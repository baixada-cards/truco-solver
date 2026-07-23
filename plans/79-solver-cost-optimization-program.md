# Plan 79 — Solver cost optimization program

Owner: Codex task `019f6bdd-035c-7c92-95fc-bd57bb5f2431`

Goal: reduce the projected cost of a useful full-game policy below $1,000,
preferably below $100, while measuring and explicitly bounding any loss of
equilibrium accuracy. The exact full-tree estimate in `SOLVER_BENCHMARKS.md`
is the baseline; no optimization gets credit from an internal estimate alone.

## Operating rules

- Each feature starts with the cheapest discriminating benchmark. Cloud spend
  is capped at **$2 per feature** until a written result justifies a larger run.
- Every approximation has a control run and reports exact exploitability or a
  clearly-labelled sampled estimate. Match value alone is not a sufficient
  correctness check.
- Preserve a lossless mode. Experimental pruning, reduced precision, and
  sampling stay opt-in until they meet their acceptance criteria.
- Record every benchmark, including negative results, in
  `SOLVER_BENCHMARKS.md`; update `RESEARCH_NARRATIVE.md` when the result changes
  the recommended solver architecture.
- Do not pay to learn something that `count-tree`, a deal-limited local solve,
  or an existing checkpoint can answer.

## Phase 1 — Policy-aware tree census (first)

Implement a space-for-time DFS counter driven by a saved policy. It must count:

1. the policy profile's supported tree;
2. player 0's unilateral best-response closure (all player-0 actions, only
   supported player-1 actions);
3. player 1's unilateral best-response closure;
4. the union of both best-response closures.

Run exact-zero and configurable probability thresholds. Missing actions when a
shallower-band policy is projected into a deeper band must use an explicit,
reported fallback (`all`, `first`, or `all-except-raise`), never a silent one.

Cheap gate: unit tests plus local `--max-deals` runs cost $0. Fetching an
existing checkpoint is allowed; no new solve is required. Proceed to a
support-restricted builder only if the BR-closure union is materially smaller
than the raw tree (initial target: <=3x the `{1,3}` tree at useful thresholds).

Result (2026-07-16): gate failed for a static all-actions BR closure. On the
actual solved 10x10/TC0/d0 policy and all 140,118 deals, exact/`1e-8` union was
the full 39.51M info sets; `1e-6` retained 39.10M and `1e-4` retained 31.27M.
Profile-only `1e-4` was 12.05M, but is not adversarially safe. Do not build the
static restricted arena. Next candidate is a space-for-time deterministic-BR
oracle census that adds only each responder's chosen action, not every action
it was allowed to consider. Measured cloud cost was approximately $0.15.
The exact BR result now exposes its already-computed `chosen_actions` vector
(with a focused ownership/action-range regression test). The corresponding
three-path arena traversal now counts `profile U chosen-BR0 U chosen-BR1` and
is regression-tested between profile and the old all-action union. The
production 10x10 census retained 12.06M info sets / 74.42M nodes at `1e-4`
(3.28x / 4.73x below raw) and added only ~8k rows beyond profile. That clears
the fixed-path gate, but not yet the builder gate: a restricted solve can
cross-combine both players' retained actions. `chosen-br-closure` now measures
that actual local action-set arena. Its production `1e-4` result is 12.149M
info sets / 74.994M nodes, only 0.76% above fixed paths and 3.25x / 4.70x below
raw. The builder gate passes. Restricted iteration must still repeat full exact
BR until gain <=0.01.

## Phase 2 — Generic neighboring-score warm starts

Add direct same-key transfer for score states in the same ladder band. Transfer
regrets, reset the cumulative average and iteration counter, and retain the
existing mão-de-onze history-remapping mode.

Cheap gate: on a deal-limited pair of adjacent scores, compare cold and warm
runs to the same exact exploitability. Require statistically meaningful fewer
iterations or wall time with no worse final exploitability. Then run at most
one right-sized spot scout, hard-capped at $2.

Result (2026-07-16): gate passed at 300 strided deals, TC0/d0, adjacent 7x7 ->
8x8 in the `{1,3,6}` band, with nonuniform symmetric continuation values. Warm
reached epsilon<=0.01 in 10 iterations / 2.7s versus cold 90 / 7.4s; at 100
iterations exploitability was 0.000172 warm versus 0.006642 cold. The source
state is itself required by the dynamic program, so its solve cost is reusable.
The all-deal production gate also passed, but with a smaller honest effect:
actual solved 10x10 -> 10x9 TC0/d0 reached epsilon<=0.01 in 40 iterations /
28m03s versus cold 90 / 45m13s, a 2.25x iteration and **1.61x wall** win. The
worker cost roughly $0.40 and was deleted. Use 1.61x in budgets. `solve-tc` now
uses disk-backed transfer too; a byte-identical local regression confirms that
releasing the source table before CFR allocation changes no result.

## Phase 2b — Restricted warm solve + exact-BR oracle

Result (2026-07-16): implementation and safety gate pass; the strict production
speed gate fails. A same-score 300-deal stress test deliberately
showed why the global audit is mandatory: internal epsilon 0.00515 audited at
0.01528 in round zero; two monotone exact-BR action additions certified 0.00671
with 9.89x fewer infos and 1.95x fully charged speedup. In the usable neighboring
7x7 -> 8x8 workflow, retained-action regret transfer certified round zero at
0.0016615, but the source closure was only 1.57x smaller and the extra audit
made it 19% slower than full warm. The <=$2 actual 10x10 -> 10x9 scout then
certified 0.009965 after three oracle rounds, retaining 2.64x fewer final info
sets, but took 41m54.5s versus 28m02.9s full warm (49% slower) and used more
peak memory. Repeated CFR restarts plus 537s of audits erased the arena gain.
Keep the safe experiment surface; do not deploy it or assign a strict-0.01
multiplier. Target 0.011 would accept round zero at 0.010560 and provisionally
~1.35x speed, but that single no-output spot is only a future approximation
gate, not budget credit.

## Phase 3 — Relaxed target and global error budgeting

Reprice the lattice at local `epsilon = 1e-2`. On the solver's `[-1, 1]`
utility scale this is at most `0.5` match-win-probability percentage points of
average unilateral best-response gain for a two-player zero-sum subgame
(`epsilon * 50`). It is not "the bot loses 0.5pp at every decision" and local
subgame bounds do not simply add; the assembled match needs a separate global
best-response evaluation. The two unilateral gains sum to `2 * epsilon`, so
one asymmetric side can leave as much as `1.0pp` at `epsilon=1e-2`; record both
components rather than presenting only their average.

Cheap gate: reuse recorded 10x10 trajectories first. Run new scouts only where
the deeper ladder changes convergence behavior, capped at $2 per tier. Produce
both the raw uniform-epsilon price and a reach/global-BR-guided allocation.

## Phase 4 — Safe lazy and regret-based pruning

Pruning begins only after a warm-up (or a validated warm start). Temporarily
skip strongly negative-regret traverser actions, but revisit them on a fixed
schedule so a stale early estimate cannot remove a later profitable deviation.
Opponent exact-zero pruning remains unchanged.

Cheap gate: deterministic small/deal-limited A/B against unpruned SyncCFR+.
Require matching exploitability trajectory within floating-point/reordering
noise and a measured traversal-time reduction. One <=$2 full-size scout only
after the local gate passes.

Result (2026-07-16): implemented opt-in SyncCFR+ pruning with an unclamped
`f32` shadow regret, explicit warm-up, paired-player full revisits, and fresh
warm-up after resume. On the same 300-deal warm control, revisiting every 2 CFR
rounds kept all 10-iteration exact-epsilon checkpoints within `8e-6` absolute
of unpruned and reduced total wall 8.6s -> 8.2s. Revisit every 10 rounds was
faster by only another 0.1s but worsened final epsilon 0.000172 -> 0.000527.
Most importantly, the warm target crossed epsilon=0.01 at iteration 10, before
pruning began. Decision: retain as experimental tight-refinement machinery;
do not pay for a scout or enable it in the epsilon=0.01 fleet.

## Phase 5 — Deep-band mini-batched MCCFR

Upgrade the existing external-sampling implementation with persistent compact
state, mini-batched chance/deal samples, neighboring-policy seeding, and exact
or consistently sampled evaluation. The old mão-band benchmark is not the
decision point: MCCFR is intended only for trees that cannot be materialized.

Cheap gate: use a full-ladder score with a bounded deal/sample/time budget on
local hardware, comparing exploitability improvement per dollar-equivalent
CPU second against full CFR on a smaller matched control. Any cloud scout gets
a wall-clock shutdown and a $2 maximum.

Result (2026-07-16): implementation complete, gate failed. Local batch 1 vs 32
was effectively identical in speed/epsilon, and a restricted in-sample 1M run's
epsilon=0.034 did not survive held-out evaluation. The actual-policy Spot scout
(10x10 policy -> 0x0 full ladder, all 140,118 training deals) grew to 20.74M
sparse info sets / 14.3 GiB RSS at 1M samples while held-out-panel epsilon
worsened 0.226 -> 0.241. Retain experiment code, do not scale this solver path.
Cloud spend including a setup-only failed attempt was only a few cents.

## Phase 6 — Representation cleanup

Evaluate `f32` accumulators, player-split pending buffers, packed solve-time
metadata, and multi-score traversal batching independently.

Accuracy contract for reduced precision:

- identical legal actions and tree counts;
- exact-BR exploitability and game value against an `f64` control at several
  checkpoints;
- resume/checkpoint round trips;
- no material increase in final exploitability (initial tolerance: absolute
  `1e-5` on cheap deterministic cases and <=1% relative at production target);
- measured RSS and wall-time improvement before adoption.

Result so far (2026-07-16): accepted two lossless metadata/buffer cleanups.
Player-local pending-regret offsets and boxed legal-action slices preserved
byte-identical checkpoints/strategies/values and exact epsilon. The combined
829k-row run measured 1.403 -> 1.399 GB peak RSS with no speed win; its fixed
structural metadata saving projects to 316 MiB on the 39.51M-row 10x10 job.
Rejected inline `SmallVec<[Action;8]>` after it increased RSS to 1.483 GB.
Reduced-precision accumulators are still pending as a separate opt-in A/B.

## Phase 7 — Ex-ante dominance pruning (proof required)

The proposed hand-strength dominance pruning is plausible engineering but
unlikely to have many safe rules in a bluffing game. A weak hand can raise as
part of an equilibrium bluffing range, so "low equity" is not dominance.

Only implement a candidate rule after writing a strategy-independent payoff
argument showing one action is never better against any opponent strategy.
Then solve a cheap fully-checkable tier with and without the rule and compare
exact best response, match value, and strategy support. Empirical similarity
alone does not prove dominance; it only guards the implementation of an
already-proven rule.

Cheap gate: proof review + local small-tier A/B + `count-tree`; $0 expected.

Review result (2026-07-16): reject the attachment's general hand-strength/
equity heuristic. Bluff raises make low equity strategy-dependent, not
dominated. The only credible hand-dependent family is a mechanically certified
forced-result rule: e.g. accept is dominated by fold only if every hidden deal
consistent with the information set and every continuation proves the hand
cannot avoid losing more than the current fold stake. That needs a universal
state-space certificate, not an equilibrium sample, before any builder rule.
The focused follow-up did find two narrow proofs and implements them in
`TraversalState::abstract_legal_actions()`:

1. The second mover's face-down play in rounds 2/3 is weakly dominated by
   showing the same card. The play resolves the round, so there is no later
   response to influence; if both versions lose round 2, the hand ends too.
   A final-round leader's hide is removed only when the responder cannot raise.
   It deliberately remains available when revealing can change the responder's
   raise range. In mao de onze, the only strategically live hiding family left
   is therefore the round-2 leader; blanket "never hide at 11" is unsafe.
2. Calling a final-round raise is removed when the caller lost round 1, led
   either hidden or the globally weakest card in round 3, and thus loses after
   accepting against every possible responder card (including the round-1
   tie-break). Fold loses only the old stake. A legal re-raise remains because
   it can still win through fold equity; at 9x9 the existing match-deciding
   raise rule removes that bluff too, leaving exactly Fold.

Unit tests pin every boundary above. A 300-deal warm A/B retained essentially
the same exact value/exploitability while reducing 829,386 to 628,742 info sets
and wall time from 8.6s to 6.1s on a noisy local run. Across all deals, raw
nodes shrink 2.26x at mao and 1.76-1.96x through the ladder tiers, but
overlapping histories leave only a 1.07-1.08x info-set shrink. The production
epsilon=0.01 warm scout reached 0.009622 in the same 40 iterations, yet took
30m49s / 61.5 GiB peak versus 28m03s / 55.7 GiB before. The branches were
largely already skipped by exact-zero CFR pruning, and table overlap dominates
VmHWM. Keep the rules; assign no cost multiplier.

Portability caveat measured 2026-07-17: the prunes are lossless only for
fresh or warm-started solves on the pruned tree. Checkpoints solved before
them keep ~36% average mass on hidden plays at affected rows and evaluate at
~0.19-0.22 (300-deal 8x8) / 0.0163 (all-deal 10x10) when projected, however
converged they were on their own tree. Treat pre-dominance artifacts as
requiring a warm re-solve before any current-tree certification or export.

The observable-card-multiplicity extension was tested and rejected at the $0
gate. Exact 300-deal censuses were unchanged at both full ladder (0x0/TC0/d0:
7,905,387 nodes / 3,859,486 info sets) and `{1,3}` (9x9: 374,605 / 193,769).
This is structural, not just an unlucky panel: `Plain(1)` would require every
remaining `Plain(0)` copy to be visible in the caller's hand/public plays, but
then the required lose-round-1/win-round-2 history cannot occur; higher ranks
require exhausting more lower cards than the five observable card slots can
hold. The existing globally weakest-card rule already covers every reachable
member of this proof family. Do not add multiplicity machinery.

Asymmetric raise pruning ACCEPTED (2026-07-17). The deployed raise prune gates
on `min(score)`; the proof works per-acting-player, so gating on the acting
player's own score removes the higher-scored player's dominated raises in
lopsided states (a strict superset of the deployed prune). Full Phase-7 gate
passed: proof (see `TreeRules::AsymmetricRaisePrune`), count-tree A/B
(symmetric cells exactly unchanged, 9×0 → 0.115× info sets), and value A/B
(9×6 Δ4.4e-7, 9×0 Δ6.1e-6, 11×6 identical — all within convergence noise).
Whole-grid tree cost falls to 0.38–0.53× (full-ladder tier, 89% of cost, to
0.36–0.51×), moving the ~$31K bracket toward ~$12–16K. It shrinks ~48 of 57
full-ladder cells; the ~9 symmetric-deep cells still need the giant worker.
Opt-in `--asymmetric-raise-prune`; a production tier scout is the next gate
before fleet credit. Details in `SOLVER_BENCHMARKS.md` 2026-07-17.

## Phase 8 — Cheap global allocator and compact exact BR

Whole-match backward induction over the score DAG is cheap once each state's
transition/BR summary exists. The current expensive part is materializing every
hand tree for the two local exact-BR passes. On 10x10, measured build + source
BR time is 461.8s versus 22,932s for the 900-iteration tight solve, so applying
the present materialized evaluator across the raw cost lattice is roughly a
$10K operation before the new dominance savings. That is not an acceptable
routine allocation pass under this plan.

Run two deliberately separate gates:

- **Sampled allocator (<= $2):** use fixed strided deal panels at representative
  states in each ladder tier/TC/position, projected neighboring policies with
  explicit missing-raise behavior, and the cheap score-DAG dynamic program.
  Report sampling intervals and both unilateral gains. Its output may prioritize
  states and epsilon targets, but is not an exact certificate.
- **Compact exact-BR evaluator (<= $2 prototype):** avoid the solver arena and
  regret arrays. Assign compact info-set ids during DFS, retain only the policy,
  counterfactual aggregates and chosen BR actions, and recompute histories as
  needed by depth. First match the existing exact BR bit-for-bit on tiny and
  300-deal games, then A/B one solved all-deal 10x10 policy for value, time and
  peak RSS. Only a measured production result earns a whole-grid cost claim.

The compact path is promising because exact BR needs far less state per info
set than CFR solving, while current cloud cost is dominated by RAM-hours. It is
not yet credited: repeated DFS passes or external aggregation may trade too much
time/I/O for the memory reduction, and the full-ladder panel must measure that.

Result so far (2026-07-16): both tools are implemented. `allocation-scout`
builds deterministic strided panels, reuses sampled profile hand-outcome
kernels through the complete score DAG, and weights representative-band
one-hand deviations by profile reach. It reports both players, TC coverage,
missing-policy decisions and panel ranges; the resulting cumulative error mass
is explicitly a priority score, not exploitability/equity. A $0 TC0 donor run
with 96 deals/three panels put 32.9% of priority in the full ladder and 25.4%
in `{1,3,6,9}` under `all-except-raise`; a separate `all` fallback sensitivity
run also put a majority in the two deepest bands. That ordering is actionable,
but the absolute mass and equity are not yet stable/certified.

`compact-br` streams only average policy rows and recomputes dynamic histories
depth by depth, retaining counterfactual aggregates and chosen actions instead
of the arena. It matches the current materialized oracle on 12- and 300-deal
tests. On the actual 300-deal 8x8 checkpoint, profile/BR differences were at
most `1.85e-10`; compact evaluation took 2.791s versus 1.509s for the arena
control (1.85x slower locally).

Result (2026-07-17): the all-deal production gate ran on all 140,118
10x10/TC0/d0 deals against the actual solved checkpoint and **passes on
memory/time**: 1,414.0s single-threaded evaluation, 24m03s process wall,
**5.95 GiB peak RSS** on a 16-GiB Spot worker. Exact whole-game certification
no longer needs arena-class RAM; per-eval cost is roughly neutral versus the
materialized oracle (~3.1x slower on a ~3.8x-cheaper worker), moving the
modeled whole-grid certification pass from ~$10K to ~$8K, with the
single-threaded pass still parallelizable. Scout cost: a few cents; VM and
disk deleted after retrieval.

The same run exposed a checkpoint-portability hazard: it printed
`epsilon=0.016309` for a checkpoint certified at 0.000248, because the
2026-07-16 proof-scoped prunes changed the tree and the old policy was
silently projected. A $0 local discrimination (300-deal 8x8) showed matched
trees agree with the arena to 5e-11 while pre-dominance checkpoints evaluate
at ~0.19-0.22 regardless of their own-tree convergence: old equilibria keep
~36% average mass on hidden plays, so concealment is load-bearing and **no
local row projection restores their quality on the pruned tree**. Projection
is now explicit (`--project-dominated remap|renormalize`, default remap:
hide mass -> same-card face-up, certificate-pruned accept -> fold; never
worse, matched-tree no-op, unit-tested) and loudly reported
(`COMPACT_BR_PROJECTION` remapped/dropped mass); the arena `--control` skips
cleanly on mismatched rows instead of panicking. Certifying or exporting
pre-dominance artifacts on the current tree requires a warm-started re-solve
(measured to heal projection: 40 iterations at 10x10->10x9) or a not-yet-built
old-tree evaluation mode. The all-deal compact-vs-arena equality leg stays
open until a current-tree all-deal checkpoint exists.

## Phase 9 — Direct and streaming warm-start into dense accumulators

The proof-pruning production scout exposed a separate peak-RSS overlap. Even
after `src` is dropped before CFR buffers, the current path first constructs a
full target `StrategyTable`, then deserializes the full source `StrategyTable`,
then copies both into `DenseAccum`. The live 10x9 process settled near 16 GiB
after transfer but had already hit a 61.5-GiB high-water mark, so it still needs
the old 75-GiB worker. This is exactly the kind of representation cleanup whose
value must be measured at peak, not inferred from steady-state RSS.

Split that overlap into two independently measurable steps. First allocate
`DenseAccum` from the target's table-index-ordered metadata and copy matching
action slots from the loaded source directly into dense regrets (plus averages
for true resume/cross-turn-up modes), avoiding the complete empty target table.
Then introduce a row-streamable checkpoint representation so the source table
also need not be fully materialized. Preserve legacy remap and action-identity
projection semantics in both paths.

Cheap gate: byte-identical checkpoint/strategy/value and identical epsilon
trajectory against the current loader on tiny, 300-deal, same-band, resume,
mao-remap, and cross-turn-up fixtures. Then one <=$2 all-deal 10x10->10x9 A/B
records transfer wall, end-to-end wall and VmHWM. It earns fleet cost credit
only if the worker can actually be provisioned in a smaller memory class.

Phase 9a result (2026-07-16): direct-to-dense projection is implemented for
disk warm starts. Same-band, cross-turn-up, and mao-remap fixtures exactly match
the old table projection, and the full Rust suite passes. The production A/B
matched epsilon at every 1/10/20/30/40 checkpoint and matched the final value;
wall fell 30m49.0s -> 29m02.8s and VmHWM 61.47 -> 43.58 GiB. Provisioning 55
instead of 75 GiB makes this identical shallow run about 1.46x cheaper. The
result earns local cost credit, but not a fleet multiplier until a deep-band
memory scout passes.

Phase 9b result (2026-07-16): the positioned checkpoint format was already
row-delimited, so a validated `CheckpointStream` now feeds matching source rows
directly into dense target accumulators; legacy artifacts retain the full-loader
fallback. Same-band, cross-turn-up and mao-remap fixtures exactly match the old
table and direct-dense paths. The identical all-deal 10x10->10x9/TC0/d0 run
again stopped at iteration 40 with epsilon `0.009622` and value `0.198171`.
Source streaming held the solve phase near 16 GiB, but converting dense output
back into the returned strategy table set the real VmHWM: 33,133,128 KiB
(31.60 GiB). Wall was 30m49.4s versus 29m02.8s direct-dense. A 40-GiB worker
is still about 24% cheaper per completed shallow job than the 55-GiB direct
loader after charging the 6.1% wall regression, but this remains uncredited for
deep tiers. The next lossless representation gate is a dense-to-checkpoint/
strategy writer that avoids the final dense+hash-table overlap.

## Deliverables and decision log

- [x] Policy-aware census, CLI, tests, and first checkpoint measurements.
  (Actual 10x10 policy measured exactly; static BR-union gate failed.)
- [x] Generic same-band warm starts and cold/warm benchmark. (Direct regret
  transfer + average reset implemented, unit-tested, and passed a $0 adjacent-
score A/B; opt-in exact-band pipeline selection is implemented; the production
10x10→10x9 scout measured a 1.61x end-to-end win.)
- [x] Restricted solver and monotone exact-BR audit loop. (Local certification
  passes; the production sparse-source scout also certified but was 49% slower
  than full warm, so the strict composition is rejected.)
- [x] `epsilon=1e-2` cost table and explanation of equity meaning.
- [x] Warm-up + reversible lazy/regret pruning benchmark. (Conservative mode
  passed the accuracy gate but has zero stopping-time benefit at epsilon=0.01;
  no paid scout and no fleet enablement.)
- [x] Mini-batched deep-band MCCFR benchmark. (Gate failed; no larger run.)
- [ ] Representation A/B matrix with accuracy and RSS results. (Lossless
  pending/metadata cases complete; reduced precision remains.)
- [x] Ex-ante dominance candidates accepted with proof or rejected with notes.
- [x] Sampled whole-match reach/error allocator and uncertainty report. (First
  TC0 projected-policy panel is a prioritization result, not a certificate.)
- [x] Compact exact-BR evaluator A/B against the current oracle. (12/300-deal
  equality to 5e-11; all-deal 10x10 memory/time gate at 5.95 GiB / 1,414s;
  all-deal equality closed 2026-07-17 via --legacy-tree self-certification:
  epsilon 0.000248280 and value 0.055711349962 match the solve-era arena
  certificate to print precision.)
- [x] Direct-to-dense warm-start loader and VmHWM A/B.
- [x] Row-streamable source-checkpoint loader and source-size VmHWM A/B.
  (31.60-GiB peak; deep target tier remains unmeasured.)
- [x] Dense-direct artifact serialization and output-side VmHWM A/B (phase
  9c: byte-identical artifacts, 16.19-GiB peak, 24-GiB worker class).
- [x] Revised full-game recommendation and projected spend.

## Current recommendation and cost bracket

> **Program outcome (2026-07-17): PAUSED, pivoting to a neural policy.** The
> best whole-grid exact bracket reached is **~$12–16K** (below), still >10× the
> <$1K goal and walled by a few deep symmetric cells no lossless method shrinks.
> The decision is to invest the remaining budget in the neural approach — see
> [EXACT_SOLVING.md](../EXACT_SOLVING.md) (cold-resume record) and
> [plan 83](83-neural-policy-approach.md). The bracket history is kept below.

Cost-bracket history (spot, whole grid):
- Raw grid, tight ε≈2.5e-4: **~$505K** baseline.
- ε=0.01 relaxation: **~$50.5K**.
- + same-band warm starts (measured 1.61× wall): **~$31K** (the 2026-07-16
  evidence-based anchor).
- + asymmetric raise pruning (2026-07-17, grid 0.38–0.53×): **~$12–16K** — the
  final and lowest evidence-based bracket. The full-ladder tier (89% of cost)
  takes the biggest cut; the residual is the ~9 deep symmetric cells.

Do not launch a raw-grid solve, a static BR-union restricted builder, or a
larger MCCFR run. If the exact line is ever resumed, the open levers are in
EXACT_SOLVING.md §6 (a ≤$2 deep-tier iteration-count scout, further structural
dominance rules, reduced precision, or — the only order-of-magnitude lever —
lossy abstraction).

Proof-scoped action pruning is now on by construction but receives no additional
budget credit after its production gate. Current materialized whole-grid exact
BR is also not a cheap allocator: measured build + both response passes project
to roughly $10K across the raw lattice. Use the planned <=$2 sampled allocator
for prioritization, and require the compact exact-BR path before treating exact
whole-match certification as compatible with the sub-$1K target.

Phase 9c closes the representation chain: the identical shallow warm job
now peaks at 16.19 GiB (24-GiB worker class, 75 -> 55 -> 40 -> 24 GiB
across phases 9a/9b/9c), and whole-lattice exact certification is measured
to fit 16-GiB workers even at the deepest band. Neither is yet credited
against the deep-tier SOLVE costs, which remain the unmeasured majority of
the bracket.

Direct-to-dense plus row-streamed source loading is a measured lossless
shallow-band win. The streamed run is 6.1% slower than direct-dense because of
row matching, but peak falls 43.58 -> 31.60 GiB; at current catalog rates a
40-GiB worker is about 24% cheaper per completed shallow job than 55 GiB. It
does not change the current $31K evidence-based fleet bracket because deeper
target bands have not established the same memory class. Even a deliberately
uniform extrapolation would move the bracket only to roughly $17-18K, and the
next dense-output cleanup would still leave a large structural gap. Streaming
plus selective solve allocation is required rather than sufficient.

The restricted solver is safe but gets no strict-target multiplier. The actual
<= $2 10x10→10x9 sparse-source composition certified after three rounds, but
was 49% slower than full warm once the full build and all audits were charged.
A 0.011 certificate could provisionally accept round zero and move the bracket
toward ~$23K, at an extra 0.05pp average-equity allowance versus 0.010, but it
needs a real output-writing/deeper-band gate before credit. Without it, another
~31x is needed for <$1K (or ~310x for <$100).

## Live handoff — profile-transfer-fleet run (2026-07-16, Codex usage-limited mid-flight)

A parallel Codex session implemented the full 225-profile Study release (all
`11x0`-`11x11` x 2 dealers x TC0-8, plus `10x10` x TC0-8) on
`codex/profile-transfer-fleet` (pushed to `origin/codex/profile-transfer-fleet`,
also checked out at `/private/tmp/truco-profile-transfer-fleet`; no PR yet).
That session hit its own OpenAI Codex usage cap mid-run (not a GCP/cost
limit) and handed off; this section is the verified resume point for whoever
continues it. Everything below was independently confirmed against live
`gcloud`/`gcloud storage` state, not just taken from the handoff message.

**Commits on the branch** (on top of `894a753`, i.e. one commit behind
`main`'s `5107047` — see the merge gotcha below): `846e97c` gate cross-score
profile transfer canaries; `ec28813` fix canary archive working directory;
`d252bc4` add the budgeted Study profile transfer fleet worker; `137c4d0`
gate fleet retries on completion markers; `36c3398` add the atomic Study
release assembler. Plus `crates/truco-solver/src/cfr.rs` / `bin/solve.rs`:
renames `--warmstart-cross-turnup` to the more general
`--warmstart-profile-transfer` (old spelling still accepted), allowing the
source checkpoint to differ by **score**, not just turn-up class, within the
same tree band.

**What it does:** for each of the 225 (score, TC, dealer) spots, either
export the already-refined checkpoint directly (5 spots have one), or attempt
a 90-second-capped one-iteration cross-score/turn-up profile transfer from
the nearest refined donor, self-certify the transferred profile against
raw-eps<=0.01 / purified-eps<=0.004 / weighted-mean-BR-gap<=0.25pp, and fall
back to the existing native (unrefined but fully converged) checkpoint —
**with a full real BR-gap table either way** — if the transferred profile
doesn't clear those bars. `tools/gcp/startup-profile-transfer-fleet.sh` is
the per-VM worker (idempotent via `COMPLETE.json` markers, so retries skip
finished rows); `tools/gcp/startup-profile-transfer-canary.sh` is the
one-shot pre-flight check; `tools/gcp/startup-profile-transfer-release.sh`
refuses to publish unless exactly 225 `COMPLETE.json` markers and 675
artifacts (chart/deep-chart/BR-gaps x 225) exist under
`gs://truco-solver-runs/profile-transfer-fleet-20260716/`.

**Pre-flight canary result (mixed, expected, already handled):** of 4
canaries, `11x0/d0` and `11x0/d1` (far-score donor `11x10`) PASSED at raw eps
0.000195/0.000266; `11x9/d1` (near-score donor `11x10`) also passed at raw
eps 0.00604; `11x9/d0` FAILED at raw eps 0.01951 even after a 10-minute tail
(average inertia from the donor's different terminal match values did not
resolve in the allotted iterations). This does not block the fleet — each of
the 225 jobs re-runs its own looser gate at runtime and silently falls back
to native per spot, so the mixed canary is exactly the intended "portfolio,
not faith" design point, not a blocker.

**Budget approach actually implemented:** not a live-cost poll. The fleet's
size/machine-type mix (4 Spot VMs: `c2-standard-8`, `n2-standard-8`,
`n2-highmem-4`, `n2-highmem-8` in `us-east1-{b,c,d}`) was picked so a full
6.5-hour run costs approximately the user's ~$5 target at current Spot
rates — an approximation was explicitly requested, not exact metering.
Backstops layer two independent mechanisms: an in-guest 6.5h (23400s) soft
deadline checked only between profiles (stops cleanly, uploads partial
progress), and a genuine GCE `maxRunDuration=28800s` (8h) with
`instanceTerminationAction=DELETE` as a hard backstop regardless of guest
health.

**Live state as independently verified (2026-07-16 ~18:08 PDT / 2026-07-17
01:08 UTC):** 4 Spot workers (`pt-fleet-10x10-0716`, `pt-fleet-11x4-7-0716`,
`pt-fleet-11x8-11-0716`, `pt-fleet-11x0-3-0716`) RUNNING in project
`truco-solver`, created ~17:44-45 PDT (~25 min into their 6.5h window at
check time). 25/225 `COMPLETE.json` markers present under
`gs://truco-solver-runs/profile-transfer-fleet-20260716/audit/`, covering
`10x10-tc0-d0` and partial runs through `11x0`, `11x4`, `11x8`. A fifth,
unrelated Spot VM `truco-compact-br-tc0` (us-central1-a) is also running —
that is Phase 8's separate compact-exact-BR production gate referenced in
`5107047`'s text above, has its own tight in-guest 55-minute shutdown and no
GCE `maxRunDuration`, and is unrelated to this fleet; no action needed there.

**Merge gotcha — resolve before landing:** this branch's merge-base with
`main` is `894a753`, one commit behind `main` HEAD `5107047` ("docs(solver):
record sub-two-dollar cost gates"). That commit added real benchmark results
to this same file (the Phase 8 `compact-br` all-deal production result, the
Phase 9b `CheckpointStream` row-streaming result, the revised
`$17-18K`/`$23K` cost-bracket text above) that this branch's copy does not
have, because the branch forked before it. A naive rebase/merge will show
these as overlapping edits to the Phase 8/9, "Deliverables", and "Current
recommendation" sections. Reconcile by hand and keep **both**: `main`'s
compact-br/streaming results plus this branch's cross-score canary
narrative. `RESEARCH_NARRATIVE.md` item 26 has the identical split — the
branch's rewrite of item 26 there replaces content that `5107047` also
changed — so check that file the same way before merging, and
`SOLVER_PLAN.md`'s "Remaining" checklist for the same re-checked/unchecked
items.

**Next steps once budget/completion is reached:** delete all `pt-fleet-*`
VMs and their boot disks; confirm 225/225 completion markers; run
`tools/gcp/startup-profile-transfer-release.sh` (it hard-refuses on anything
less than 225 complete + 675 artifacts) to cut the immutable public release;
rebase the branch onto current `main` and reconcile the merge gotcha above;
merge; update production and deploy per `DEPLOYMENT_PLAN.md`.

**Status update (2026-07-17 01:37-02:00 UTC, Claude Code session):**
`pt-fleet-11x4-7-0716` was Spot-**preempted at 2026-07-16 18:12:34 PDT**
(~28 min in) and, per its `instanceTerminationAction=DELETE`, no longer
exists. Verified via `gcloud compute operations list`
(`compute.instances.preempted`). Before dying it completed the 12 rows
`11x4` TC0-TC5 x both dealers; `11x4` TC6-8 and all of `11x5`-`11x7`
(~44 rows) are orphaned until a replacement runs that band's manifest.
The other three workers are RUNNING and progressing normally: 47/225
`COMPLETE.json` markers at 01:37 UTC (was 25 at 01:08). The unrelated
`truco-compact-br-tc0` VM has self-terminated as designed. A replacement
worker (`pt-fleet-11x4-7-0716r2`, same config: Spot `n2-standard-8`,
120GB pd-balanced, fresh 6.5h soft / 8h hard deadlines, manifest
`profile-jobs-11x4-7.tsv`, code archive `acf29e3` — identical to the
surviving siblings' verified metadata) was prepared by the session and
launched by the user at ~02:07 UTC after the local permission classifier
blocked agent-side VM creation. Verified RUNNING in `us-east1-b` with the
correct manifest and 8h/DELETE backstop; serial console confirms the
startup script executing. The worker is marker-idempotent, so it will
skip the 12 finished `11x4` rows and run only the orphaned ~44.

**Fleet COMPLETE (2026-07-17 05:23 UTC):** 225/225 `COMPLETE.json`
markers and exactly 675 artifacts verified. All four workers exited
`status=COMPLETE` per their `worker-summary.json` (10x10: 9 rows in
4h34m; 11x0-3: 72 in 3h36m; 11x4-7 replacement: 72 in 3h01m; 11x8-11:
72 in 3h18m) and self-powered-off.

**Per-spot outcome (aggregated from all 225 markers):** 192 spots (85%)
shipped the transferred profile, all inside the certification bars —
worst raw eps 0.00627 (gate 0.01), worst purified eps 0.00395 (gate
0.004), worst weighted-mean BR-gap 0.098pp (gate 0.25pp); medians
0.00124 / 0.00046 / 0.014pp. 5 spots exported their pre-existing
refined checkpoints. 28 spots (12%) hit `FALLBACK` and shipped the
native converged checkpoint (median raw eps 0.00001, worst 0.0168)
with full BR-gap tables — the intended portfolio behavior. The
fallback markers' metrics describe the SHIPPED native profile (the
worker re-runs its export on the native strategy after rejecting the
transfer, `startup-profile-transfer-fleet.sh:154-159`, and the marker
reads the regenerated chart/brgap summary). So the striking pattern is
real: those 28 native profiles pair near-zero whole-game raw eps
(median 0.00001) with large weighted-mean per-decision BR-gaps (median
2.55pp, max 12.59pp) — global eps is reach-weighted and blind to
off-path decision quality, which is also plausibly why the transfer
gate failed on exactly these spots. The rejected transfer candidate's
own summary is preserved separately as `transfer-brgap-summary.json`
in each spot's audit prefix if a comparison is ever wanted.

**Cleanup (2026-07-17, no action needed):** GCE's `maxRunDuration=8h` +
`instanceTerminationAction=DELETE` backstop fired as system-initiated
`compute.instances.deferredDelete` on all four VMs (08:44 UTC for the
three originals, 10:08 UTC for the replacement), after each guest had
already powered off; boot disks auto-deleted. Verified: zero instances
and zero disks remain in `truco-solver`. The 8h hard backstop doubles
as automatic cleanup — deliberately reusable pattern.

**Release (user-approved 2026-07-17):** id `20260717-full-225-v1`
(existing releases: `20260716-bootstrap-a8ffdad`,
`20260716-stealth-v1`), published by a one-shot `pt-release-0717`
assembler VM running `tools/gcp/startup-profile-transfer-release.sh`.
Published and verified 2026-07-17 15:56 UTC: assembler exited
`COMPLETE`, public manifest serves 225 spots / 675 sha256-manifested
artifacts (7.60 GiB). The branch rebase/merge with the merge-gotcha
reconciliation was completed separately (branch tip `36c3398` is fully
contained in `main`; both sides of the overlapping doc edits verified
present). Deploy: local `.env` now pins `STUDY_MANIFEST_URL` to
`20260717-full-225-v1` and `docs/ops/study-lab-deployment.md` records
the release; deployed 2026-07-17 with user-authorized Droplet SSH
(`make prod-env-sync`): the live stealth route now pins
`20260717-full-225-v1`, `X-Robots-Tag: noindex, nofollow` intact, CORS
and immutable cache headers verified on the new artifacts. This closes
the profile-transfer-fleet arc end to end. `STUDY_LAB_MODE` stays
`stealth` — going `public` is a separate product decision.

## Phase 9c — Dense-direct artifact serialization (2026-07-17)

Both remaining output-path overlaps are removed losslessly. The checkpoint
writer no longer materializes owned rows plus a whole-file byte buffer (it
sorts a borrowed row index and streams through a BufWriter), `solve-tc`
writes the average-strategy artifact directly from dense accumulators, and
`skip_return_table` drops the end-of-solve hash-table rebuild. Unit test
pins byte-identity to the historical layout; a deterministic 300-deal
re-solve reproduces pre-change strategy/checkpoint/gv artifacts bit-for-bit.

Result (2026-07-17): the <=$2 production A/B passed. The identical all-deal
10x10 -> 10x9 warm solve reproduced the epsilon trajectory at every
checkpoint and finished at **16.19 GiB peak** (vs 31.60 GiB streamed /
43.58 direct-dense / 61.47 original) in 17m56.8s wall on a 24-GiB Spot
worker (~$0.06 scout). Conservative credit: the 40->24 GiB class ratio
(~1.7x cheaper per completed shallow job) at equal wall; the additional
~1.7x per-iteration wall win observed is likely host variance and is not
credited. Deep-tier certification was subsequently measured (compact BR fits
8.13 GiB at 0×0 — see "Follow-up instruments" below and SOLVER_BENCHMARKS
2026-07-17); deep-tier SOLVE memory remains unmeasured (the probe
under-provisioned and livelocked).

## Follow-up instruments landed 2026-07-17 (post compact-BR gate)

- `compact-br --legacy-tree` (TreeRules) evaluates pre-prune artifacts on
  their own tree; validated to reproduce solve-era certificates exactly.
  Production self-cert PASSED (epsilon 0.000248280 vs solve-era 0.000248,
  value match to 12 digits); clean 10x10-as-is transfer certificates at
  10x9/9x10 under `legacy-cert-v1` (~$0.10).
- `compare-policies [--remap-turnup] [--reach-weighted] [--legacy-tree]`:
  descriptive similarity of neighboring equilibria. Mão-band result:
  ~94% row-identical unweighted, 99.99% pure-row agreement, but
  reach-weighted mean TV 0.129 with only half the play mass on
  near-identical rows — transfer economics are decided by a small, shallow,
  high-reach, score-sensitive core.
- Deep-band certification scout (`deep-br-0x0-v1`): COMPLETE. Whole-game
  exact BR of the full 0x0 ladder over all deals fits in **8.13 GiB peak**
  (2h40m single-threaded, 14.6B DFS visits, BR depth 13) — certification
  memory scales with info sets (2.1x rows -> 1.4x RSS), not nodes (8.2x),
  so a 16-GiB worker class covers the whole lattice's certification passes.
  Its value output was void (incomplete match-value table; no complete DP
  table exists out of order — see the 2026-07-17 benchmark entry), which
  produced two guardrails: compact-br aborts on unsolved successor cells,
  and the mv(10,10)=0.5 footnote now annotates the 10x9 scout family.

## Deep-band solve memory: under-provisioned probe (2026-07-17, negative)

The `deep-solve-0x0-v1` scout that was meant to replace the modeled deep-tier
solve memory with a measurement instead livelocked: RSS reached 148.5 GiB in
43 min of tree build (still climbing) on a no-swap 160-GiB box and never
completed iteration 1. Root cause was a sizing error — the box was set at the
top of the same unverified model the probe was testing, with no headroom, and
no swap/clean-abort so it thrashed silently rather than erroring. Lower-bound
result stands (deep 0x0 solve build+start > 150 GiB), and any re-run must (a)
use a >=256 GB class worker and (b) add a fail-fast RSS guard or swap so a
mis-size fails in minutes. Recorded in SOLVER_BENCHMARKS 2026-07-17.
