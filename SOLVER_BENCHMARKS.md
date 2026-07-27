# Truco Solver Benchmarks

Tracking performance metrics across development for the paper.

---

## 2026-07-16 — Policy-aware DFS census scaffold + epsilon=1e-2 repricing

Implemented the first gate in `plans/79-solver-cost-optimization-program.md`:
`solve count-tree` can now count a fixed policy's supported profile, either
player's unilateral best-response closure, or the union of both BR closures.
It is still a space-for-time DFS: no tree arena or regret table. Support can use
average or current regret-matched probabilities, configurable thresholds, and
an explicit missing-action fallback for projecting a shallower policy into a
deeper raise band. A streaming compact loader reads a strategy artifact without
materializing its serialized entry Vec, full `InfoSet` map, or regrets.

Correctness tests establish that (1) a full-support policy-profile count exactly
matches the raw builder/count-tree nodes and info sets, (2) profile <= each BR
closure <= BR-union <= raw, (3) projected missing raises obey the explicit
fallback, and (4) an above-maximum threshold still keeps the argmax action.

### First local structural baseline (not an equilibrium result)

Score 0x0, TC0, dealer 0, the same 1,000 strided deals for every row. The seed
policy was intentionally empty with `all-except-raise` fallback: all card and
response actions remain supported, but previously unavailable raises are not
seeded. This validates the deeper-band projection plumbing and measures the
most optimistic no-new-raise skeleton; it must NOT be described as the support
of the solved 10x10 equilibrium. Cost: $0, 21.2s raw + 2.3s for all four policy
closures on the local machine.

| closure | nodes | distinct info sets | shrink vs raw nodes / info sets |
|---|---:|---:|---:|
| raw full ladder | 47,941,043 | 16,510,497 | 1.0x / 1.0x |
| profile support | 354,712 | 119,940 | 135.2x / 137.7x |
| BR0 closure | 1,440,215 | 488,441 | 33.3x / 33.8x |
| BR1 closure | 1,434,123 | 490,090 | 33.4x / 33.7x |
| union(BR0, BR1) | 2,519,626 | 858,591 | **19.0x / 19.2x** |

The BR union is the relevant architecture gate: profile reach alone omits the
responder's profitable deviations and would overstate the safe reduction by
~7x here.

### Exact actual-policy census: static BR closure is not enough

The decision run used the existing solved 10x10/TC0/dealer-0 average strategy
(39,508,752 entries; 5.04 GiB), all 140,118 deals, and `missing-policy=all`.
No solve was repeated. Raw is 352,297,258 nodes / 39,508,752 info sets. Counts
below are exact; percentages are reductions from raw.

| threshold | profile nodes / infos | BR0 nodes / infos | BR1 nodes / infos | union nodes / infos |
|---:|---:|---:|---:|---:|
| exact `>0` | 236.92M / 33.97M | 236.92M / 33.97M | 352.30M / 39.51M | **352.30M / 39.51M** (0% / 0%) |
| `1e-8` | 234.34M / 33.60M | 234.34M / 33.60M | 352.30M / 39.51M | **352.30M / 39.51M** (0% / 0%) |
| `1e-6` | 210.34M / 30.62M | 229.96M / 33.01M | 321.53M / 36.71M | **341.15M / 39.10M** (3.2% / 1.0%) |
| `1e-4` | 74.37M / 12.05M | 146.98M / 21.65M | 172.45M / 21.67M | **245.06M / 31.27M** (30.4% / 20.9%) |

At exact support, player 0 retains positive probability on every action needed
to expose player 1's entire response tree, so BR1 and the union are raw. Even
the aggressive `1e-4` threshold shrinks the union only 1.26x by info sets and
1.44x by nodes. The plan-79 gate therefore FAILS: do not build a static arena
containing every action in both unilateral-response closures, and do not use
the 3.28x-smaller profile count as a cost claim without an adversarial check.

This does not rule out a double oracle. The closure counter deliberately keeps
ALL actions for the responding player; an actual best response chooses one
action per information set. A stronger next census should compute the two
deterministic best-response policies space-for-time and count
`profile U chosen-BR0 U chosen-BR1`, adding only the selected deviations and
iterating until no profitable oracle action remains. That retains an exact
global-BR stopping test without permanently allocating every candidate action.

The cost-capped worker took 54m24s, used 4,632,884 KiB peak RSS, and ran on one
`n2-highmem-8` Spot VM: approximately $0.15 compute plus pennies of disk. A
first setup attempt failed before counting because GNU `time` was absent; its
four VM-minutes and logs were retained, and total feature spend remained far
below $2.

### Chosen-BR union counter: exact oracle plumbing complete

The exact backward-induction result now retains the chosen action index at
every responder-owned information set. A new `chosen-br-union` census builds
the full tree once, aligns the saved policy, computes both exact best responses,
and then walks the arena with three path bits: profile, chosen BR0, and chosen
BR1. It therefore counts `profile U chosen-BR0 U chosen-BR1`, rather than the
much wider closure containing every legal responder action. The count still
includes the profile explicitly because a double-oracle seed must preserve its
existing mixed actions, not replace each player by a pure best response.

The $0 three-deal control (10x10/TC3/d0, deliberately undertrained two-iteration
checkpoint) retained 741 nodes / 297 info sets. Focused tests establish
`profile <= chosen-BR union <= all-action BR union`, validate chosen-action
ownership/ranges, and exercise the CLI through both exact and `1e-4` support.

The production decision run reused the solved 10x10/TC0/d0 profile and exact
match values over all 140,118 deals. The chosen union is extremely close to
profile support: exact BR adds only 4,279 info sets at threshold zero, 8,425 at
`1e-6`, and 8,315 at `1e-4`.

| support threshold | chosen-union nodes | chosen-union infos | shrink vs raw nodes / infos |
|---:|---:|---:|---:|
| exact `>0` | 236,948,701 | 33,977,732 | 1.49x / 1.16x |
| `1e-6` | 210,404,830 | 30,626,840 | 1.67x / 1.29x |
| `1e-4` | **74,424,921** | **12,058,024** | **4.73x / 3.28x** |

This passes only the fixed-PATH-union gate at aggressive support pruning. It is
not yet the arena size of a restricted re-solve: when both players retain
profile plus BR actions, the solver may cross-combine actions from the three
paths. A new `chosen-br-closure` count measures that local action-set closure
and must pass before a builder. Even then, chosen actions are exact BRs only to
the current full profile; after re-solving, new deviations may appear. The safe
design must iterate restricted solve -> fresh full-game exact BR -> add chosen
deviations until global gain is <=0.01. The fixed-path union is an optimistic
lower bound, not a certificate or final cost multiplier.

The follow-up production closure rerun resolves that arena-size ambiguity:

| threshold | action-closure nodes | action-closure infos | overhead vs fixed paths | shrink vs raw nodes / infos |
|---:|---:|---:|---:|---:|
| exact `>0` | 239,591,252 | 34,300,783 | 1.11% / 0.95% | 1.47x / 1.15x |
| `1e-6` | 212,799,711 | 30,941,226 | 1.14% / 1.03% | 1.66x / 1.28x |
| `1e-4` | **74,993,779** | **12,149,296** | **0.76% / 0.76%** | **4.70x / 3.25x** |

Cross-combinations are therefore small, not an explosion. The `1e-4` closure
passes the first-arena builder gate. This remains conditional evidence: oracle
rounds can add actions, and the threshold itself has no standalone error bound;
only the repeated full-game BR <=0.01 test can certify the final restricted
profile. The rerun took 11m49s, again peaked at ~28.1 GiB, and kept cumulative
chosen-BR feature compute below roughly $0.50. Its artifacts are under the
same prefix with suffix `10x10-tc0-d0-action-closure`; its VM/disk were deleted.

The exact pass reported BR totals `0.055951662` and `-0.055455101` (average
exploitability `0.000248281`, consistent with the solved checkpoint). Full-tree
build + two BR passes + three counts took 11m18s and peaked at 29,450,448 KiB
RSS on one 75-GiB Spot worker; whole-VM lifetime was about 15 minutes and
estimated compute remained below $0.25. Artifacts are under
`gs://truco-solver-runs/cost-opt-20260716/chosen-br-census/10x10-tc0-d0/`;
the VM and disk were deleted.

### Restricted solve + monotone exact-BR oracle prototype

The closure is now a solver-ready compact arena: table indices and legal-action
metadata are remapped, retained actions keep their original identity, and a
same-band warm start can transfer regret for an action SUBSET rather than
requiring the original full row. `restricted-bench` repeatedly solves that
arena, maps the result back to a complete full-tree profile, runs both exact
full-tree best responses, and monotonically adds every newly chosen responder
action. An internal restricted epsilon never substitutes for the full audit.

The first $0 release-mode test used 300 strided TC0/d0 deals at target 8x8.
With a circular already-solved 8x8 policy only as a builder/certificate stress
test, round zero missed the global target despite internal epsilon 0.00515;
the full audit correctly reported 0.01528. Two oracle additions certified:

| round | restricted nodes / infos | shrink vs full nodes / infos | added next | internal eps | full audit eps |
|---:|---:|---:|---:|---:|---:|
| 0 | 100,757 / 42,802 | 23.05x / 19.38x | 3,356 | 0.005146 | 0.015284 |
| 1 | 191,696 / 77,954 | 12.11x / 10.64x | 497 | 0.005809 | 0.011287 |
| 2 | 208,112 / 83,901 | 11.16x / 9.89x | — | 0.006222 | **0.006709** |

Charging full-tree build, initial source BRs, every restricted rebuild/solve,
and all three full audits gives 3.907s versus 7.601s for full-tree CFR to the
same target: **1.95x end-to-end**, not the misleading sweep-only number.

The usable workflow starts from a required NEIGHBOR, so a second $0 test used
the 7x7 checkpoint to solve 8x8 and seeded both full and restricted CFR with
those regrets. The 100-iteration source was still broad at threshold `1e-4`:
the initial arena shrank only 1.57x by infos. Both paths reached epsilon in 10
iterations; restricted round zero audited at 0.0016615, but setup + its extra
audit took 3.313s versus 2.794s full warm (**0.84x**, a 19% slowdown). Therefore
do not apply the same-score closure factor to the budget.

The <=$2 production composition scout then used the actual converged
10x10/TC0/d0 source to solve all 140,118 deals at 10x9/TC0/d0. It charged the
full target-tree build, source BR closure, every restricted rebuild/solve, and
every exact full-tree audit:

| round | restricted nodes / infos | shrink vs full nodes / infos | added next | internal eps | full audit eps | cumulative restricted end-to-end |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 76,922,753 / 12,439,903 | 4.580x / 3.176x | 6,236,142 | 0.008082 | 0.010560 | 1,132.1s |
| 1 | 100,005,963 / 14,689,817 | 3.523x / 2.690x | 87,507 | 0.009961 | 0.010126 | 1,771.3s |
| 2 | 102,931,767 / 14,951,269 | 3.423x / 2.643x | — | 0.009965 | **0.009965** | 2,399.9s |

The command certified after three rounds, but took **41m54.5s** and peaked at
64,217,144 KiB RSS. The production full-tree warm control took 28m02.9s and
58,372,788 KiB, so restriction was **0.67x as fast (49% slower)** and used
more memory in this benchmark implementation. The exact audits alone cost
537.1s; restarting CFR after each oracle growth erased the smaller-arena gain.
Do not deploy this strict restricted-game composition or assign it a fleet
multiplier. Artifact:
`gs://truco-solver-runs/cost-opt-20260716/restricted/10x10-to-10x9-tc0-d0/`;
the Spot VM/disk were deleted.

There is a bounded approximation lead, not a recommendation: if the certified
target were deliberately raised again to `0.011`, round zero's exact
`0.010560` would pass. Including the command's fixed load overhead puts that
path at roughly 20m47s, provisionally ~1.35x faster than the warm control. It
raises the average unilateral match-equity allowance from 0.50pp at `0.010`
to at most 0.55pp at `0.011`; versus the control's measured `0.009825`, round
zero left about 0.037pp more average match equity. It wrote no production
artifacts and is one shallow-band spot, so it receives no fleet credit.

### What epsilon=1e-2 means and the first-order price

The solver's exploitability is `(BR0 + BR1) / 2` in utility space `[-1,1]`.
Equivalently it is the AVERAGE unilateral deviation gain. Thus epsilon=0.01 is
**0.5 match-win-probability percentage points average equity left to best
response** (`0.01 * 50`), root/reach weighted across the subgame—not 0.5pp at
every decision. The two unilateral gains sum to `2 epsilon`; one asymmetric
side can be as high as 1.0pp, so future reports must expose both components.

The measured 10x10 SyncCFR+ run already reached exact epsilon=0.0095 at iter 90,
versus the new census cost anchor's 900-930 iterations / epsilon~2.5e-4. Purely
dividing the refined spot table by 10 gives this optimistic first-order view
(before fixed build/certification cost and before measuring deeper convergence):

| tier | census spot cost | epsilon~1e-2 first-order |
|---|---:|---:|
| {1,3} remaining | $54 | ~$5 |
| {1,3,6} | $4,127 | ~$413 |
| {1,3,6,9} | $53,956 | ~$5,396 |
| {1,3,6,9,12} | $447,291 | ~$44,729 |
| **total** | **~$505K** | **~$50.5K** |

So relaxing epsilon is a real ~10x first step, but cannot meet the $1K target
without structural pruning/sampling and warm starts. On the measured shallow
10x10 job, the right-sized spot estimate at epsilon~0.01 was $0.15-0.25/job,
versus ~$1.98/job in the 900-iteration census anchor (an 8-13x observed range).

### Same-band neighboring-score warm-start implementation

`--warmstart-from` now detects identical band signatures. Same-band targets
copy every exact-key/exact-action regret row, but deliberately reset cumulative
average strategy and iteration weighting to zero because the neighboring score
changes terminal match utilities. The historical 11x11 -> asymmetric mão
history-remapping warm start remains separate and retains its old semantics.
Both transfer modes have focused regression tests. `solve-tc --max-deals N`
was added as an explicitly benchmark-only, deterministic strided-deal control;
it rejects tree-cache reuse so a subset can never poison a production cache.

First $0 paired benchmark: TC0/dealer0, 300 identical strided deals, SyncCFR+,
100 source iterations at 7x7, then cold versus regret-seeded 8x8. The scores are
adjacent and share the `{1,3,6}` tree band. To avoid the misleading equivalence
created when every unsolved continuation defaults to 0.5, the control table used
nonuniform, symmetric continuation equities (for example `mv(8,7)=0.55`,
`mv(10,7)=0.68`, `mv(9,8)=0.55`, `mv(11,8)=0.70`, with mirrored complements).
These are synthetic, not production match values, but they ensure the two games
have genuinely different terminal utilities while preserving a deterministic A/B.

| target 8x8 | iter 10 | iter 20 | iter 90 | iter 100 | wall to epsilon<=0.01 |
|---|---:|---:|---:|---:|---:|
| cold | 0.366289 | 0.146124 | 0.008176 | 0.006642 | 7.4s / 90 iter |
| warm from 7x7 regrets | **0.001661** | **0.000828** | 0.000178 | **0.000172** | **2.7s / 10 iter** |

The cheap gate passes: warm starting cut iterations to epsilon=0.01 by **9x**
and measured wall time by **2.7x** even though this tiny run is dominated by
tree build, checkpoint loading, and exact-BR evaluations. At equal 100
iterations, exploitability was **38.6x lower**. The 7x7 source solve is not an
extra fleet cost because that higher state is already a required dynamic-program
predecessor; its checkpoint is reused by later same-band states. This is strong
enough to wire generic same-band predecessor selection into the pipeline before
spending on a <=$2 production-sized confirmation.

That pipeline path is now implemented behind `--warmstart-neighbors`. Candidate
selection considers only one-point-higher states already solved by the dynamic
program and requires the exact same `band_signature`; an existing target
checkpoint always resumes instead. The source checkpoint is loaded only for
the regret copy and dropped before dense CFR buffers are allocated, avoiding a
second strategy table for the duration of a deep solve. This is verified
plumbing; the cost-capped scout below supplies the production speed claim.

The production-sized gate is now complete: solved 10x10/TC0/d0 -> 10x9/TC0/d0,
all 140,118 deals, exact epsilon target 0.01, on the same 75-GiB Spot SKU as the
cost census.

| target 10x9 | iterations | final epsilon | `/usr/bin/time` wall | peak RSS |
|---|---:|---:|---:|---:|
| cold | 90 | 0.009675 | 45m13s | 37.5 GiB |
| warm from solved 10x10 regrets | **40** | 0.009825 | **28m03s** | 55.7 GiB* |

This is a **2.25x iteration** and **1.61x end-to-end wall** reduction (1.73x by
the solver's internal `Done` timer). The one-shot VM lived ~77 minutes, costing
roughly $0.40 at the recorded $0.314/hour rate—well below $2—and was deleted
with its disk. Artifacts are at
`gs://truco-solver-runs/cost-opt-20260716/warmstart/10x10-to-10x9-tc0-d0/`.
The starred peak came from the scout binary's CLI retaining the loaded source
table. `solve-tc` now uses the pipeline's disk-backed transfer and releases the
source before dense CFR allocation; a local before/after regression produced
byte-identical checkpoint, strategy, and value files.

### Reversible regret pruning: accurate but not useful at epsilon=1e-2

Implemented an opt-in SyncCFR+-only pruning mode. Because CFR+ clamps its
strategy-driving cumulative regrets at zero, the classifier uses a separate
`f32` shadow sum of UNCLAMPED instantaneous regrets. After an explicit warm-up,
it skips a traverser's action only when the ordinary current probability is
exactly zero and the shadow regret is below `-threshold`. Full-width revisits
cover BOTH alternating-player sweeps as a pair; after checkpoint resume the
unserialized shadow receives a fresh warm-up. Trembling is rejected because
its purpose is precisely to visit every action. The default/lossless path is
unchanged and allocates no shadow vector.

Same $0 adjacent-score control as above: warm 8x8 from the 100-iteration 7x7
checkpoint, TC0/d0, 300 strided deals, 100 target iterations. Threshold `1e-4`,
warm-up 10. The 1.99M action slots add a 7.6-MiB opt-in shadow.

| mode | full revisit cadence | total wall | exact epsilon at iter 100 | result |
|---|---:|---:|---:|---|
| unpruned control | every sweep | 8.6s | 0.000172 | reference |
| aggressive | every 10 CFR rounds | 8.1s | 0.000527 | accuracy gate fails |
| conservative | every 2 CFR rounds | **8.2s** | **0.000180** | abs delta `8e-6`, gate passes |

The conservative trajectory stayed within `8e-6` absolute exploitability at
every 10-iteration checkpoint and saved 4.7% total wall (roughly 6% before the
final artifact write). That is a real but small deep-refinement speedup. It
does **nothing** for the intended epsilon=0.01 pipeline: the warm start reaches
0.001661 at iteration 10, before the first pruned sweep. Therefore no paid scout
is justified and pruning should not be enabled in the relaxed-target fleet.
Keep it experimental for unusually tight refinements or a deeper band where a
warm target demonstrably remains above its stopping threshold after warm-up.

### Exact representation cleanup: safe, modest, and measured

Two lossless solve-time changes were A/B-tested against the pre-change binary:
SyncCFR+'s pending-regret array now overlays player-local offsets (only one
player accumulates regret per alternating sweep), and legal-action metadata
uses a 16-byte boxed-slice header instead of a 24-byte `Vec` header. The latter
saves exactly 8 bytes per built info set (316 MiB at the measured 39.51M-row
10x10 job) without changing treepack/band-meta wire formats. An inline
`SmallVec<[Action;8]>` candidate was also tested and REJECTED: its fixed inline
capacity increased peak RSS to 1.483 GB and wall to 5.64s on the control.

Control: the same warm 7x7 -> 8x8, TC0/d0, 300 deals, 20 SyncCFR+ iterations,
with `/usr/bin/time -l` outside the sandbox. The accepted representation wrote
checkpoint, strategy, and game-value files byte-for-byte identical to control
(checkpoint SHA-256 `6d17ccd8...cc57`); exact epsilon was 0.242129/0.001661/
0.000828 at iterations 1/10/20 in both. Peak RSS was 1,402,617,856 bytes before
versus 1,399,193,600 bytes in the combined accepted run (allocator-sensitive;
the pending-only run measured 1,389,297,664). Wall was 4.44s vs 5.42s in the
single combined run, so there is no demonstrated speedup. Keep the exact
memory cleanup, but assign it no cost multiplier; reduced-precision
accumulators remain a separate opt-in A/B requiring non-bit-exact tolerances.

### Seeded mini-batch MCCFR: implementation works, cost gate fails

Implemented a true frozen-strategy external-sampling mini-batch path: multiple
deal samples accumulate sparse pending regret/average updates and fold once per
batch, the sparse table persists across chunks, and an existing compact policy
can seed new rows' current strategy with pseudo-regret while cumulative average
starts fresh. Evaluation uses a fixed mid-stride panel disjoint from a
restricted training prefix. Focused tests cover seeding, persistent updates,
and the existing MCCFR behavior.

The $0 local controls separated the effects:

| setup | samples | batch | held-out/exact-panel epsilon | train time | info sets |
|---|---:|---:|---:|---:|---:|
| 300-deal 7x7 seed -> 8x8, train/eval on same 300 (optimistic) | 100k | 1 | 0.14720 | 11.4s | 0.897M |
| same | 100k | 32 | 0.14816 | 11.3s | 0.892M |
| same, disjoint 100-deal mid-stride eval | 100k | 1 | 0.33408 | 11.3s | 0.897M |
| same, disjoint eval | 100k | 32 | 0.33474 | 11.5s | 0.892M |
| full 140,118-deal training distribution, partial 300-deal seed | 100k | 32 | 0.51461 | 22.2s | 8.733M |

Batch 32 is therefore neither faster nor more accurate than batch 1. A 1M
restricted/in-sample run reached epsilon 0.03413, but that number is explicitly
NOT decision-grade because its evaluation deals were in the tiny training
support. On a smaller materializable 100-deal control, full SyncCFR+ reached
epsilon 0.13288 in 20 iterations / 7.2s and remained preferable.

The production-shaped Spot scout projected the complete solved 10x10 policy
into the full-ladder 0x0/TC0/d0 game, sampled across all 140,118 deals, and
evaluated a fixed 100-deal mid-stride panel:

| samples | held-out-panel epsilon | sparse info sets | cumulative train time |
|---:|---:|---:|---:|
| 250,016 | 0.226361 | 6.96M | 47.5s |
| 500,032 | 0.230844 | 12.06M | 96.2s |
| 750,048 | 0.238028 | 16.58M | 147.6s |
| 1,000,000 | **0.241113** | **20.74M** | **196.7s** |

The command took 4m19s including four exact-panel evaluations and peaked at
14,971,276 KiB RSS. The seed helps the shared shallower histories but cannot
cover new deeper-raise histories; resetting stale average mass means those
newly sampled rows dominate the learned average. Quality did not approach
`1e-2` and worsened over this budget while the supposedly sparse state grew
rapidly. Decision: retain the implementation as an experiment, but reject
mini-batched seeded MCCFR as the current deep-band plan and spend no more on
it. One failed four-minute setup attempt used a mistyped GCS fixture name and
ran no samples; attempt 2 completed. Total cloud spend was only a few cents,
far below the $2 feature cap.

### Revised budget after the cheap gates

The $505K raw-grid baseline already benefits from CFR's exact-zero opponent
branch pruning; it does not include a 17x sustained-tremble surcharge. For this
exercise, omitting sustained trembling prevents that surcharge but cannot be
claimed again as a new speedup. Epsilon=0.01 alone gives the measured/first-
order ~$50.5K estimate. The production neighboring-score warm start cut target
iterations 2.25x and end-to-end wall 1.61x. Applying ONLY that measured wall
factor gives a current planning estimate of about **$31K**. It is still not a
fleet quote: one shallow TC proves the mechanism but deeper ladder bands can
have different convergence.

Lossless representation work gets no time multiplier; static BR union and
seeded MCCFR failed; regret pruning has no iterations to accelerate at the
warm epsilon target. The restricted solver's exact oracle composition is also
rejected: the actual sparse-source 10x10->10x9 scout certified, but took
41m54.5s versus 28m02.9s for full warm. The current evidence-backed strict
`epsilon=0.01` budget therefore remains **~$31K** and needs another ~31x for
<$1K (or ~310x for <$100). A deliberately looser `epsilon=0.011` could have
accepted restricted round zero and provisionally suggests ~$23K, but that
single no-output scout is not sufficient evidence for a fleet multiplier.

---

## 2026-07-15/16 — Exact tree-size census via space-for-time DFS counting

Answered "can we measure tree size without a shitload of RAM" by trading space
for time: a plain depth-first walk over `TraversalState` (the same lightweight
struct `build_node` already used, with none of the `TreeScratch`/
`InfoSetRegistry` machinery) needs only `O(depth)` stack space regardless of
tree size, plus a `HashSet<u64>` of seen info-set keys (~14 bytes/entry
observed, vs. the solver's own ~1.9 GB/M-info-sets true footprint). New
`count_tree_size` (`game_tree.rs`) + `solve count-tree` CLI subcommand.

**Validation:** a dedicated unit test (`test_count_tree_size_matches_full_build`)
compares exact node/info-set counts against the real tree builder
(`build_all_trees_with_dealer`) across 5 scores × both dealers — bit-for-bit
match. At production scale, `count-tree --score 10x10` reproduced the
existing 39,508,752-info-set anchor exactly in 138s / ~2.25 GB RSS on a local
machine (25.77 GB RAM, 10 cores) — no GCP high-RAM VM needed to measure a tree
whose *solve* needs ~75 GB.

### Exact per-tier info-set counts (TC0, dealer 0)

| tier | ladder | info sets | nodes | growth vs. prior tier |
|---|---|---:|---:|---:|
| mão de onze | none | 5,611,123 | 49,588,022 | — |
| {1,3} (Stage B) | ≤1 raise | 39,508,752 | 352,297,258 | 7.04x |
| {1,3,6} | ≤2 raises | 129,144,643 | 1,109,988,460 | 3.27x |
| {1,3,6,9} | ≤3 raises | 341,656,035 | 2,873,154,420 | 2.65x |
| {1,3,6,9,12} (full ladder) | ≤4 raises | 812,865,845 | 6,704,430,530 | 2.38x |

**Structural finding, not just a sample:** within a ladder tier, info-set count
depends *only* on the tier (i.e. on which of `{1,3}`/`{1,3,6}`/`{1,3,6,9}`/
`{1,3,6,9,12}` applies), not on the exact score. Verified directly: 9×9 and
10×9 reproduced 10×10's 39,508,752 exactly; 6×6 matched 8×8's count at a
matched 5,000-deal subsample (11,255,396); 3×3 matched 5×5 (29,424,637); 1×1
and 2×2 both matched 0×0 (69,178,676). So the four non-mão numbers above are
exact for every state in their tier, not representative-state estimates.

### Job-count derivation (dealer symmetry)

For any score pair `(a,b)`, `mv(a,b,d) = 1 − mv(b,a,1−d)`, so each **ordered**
`(a,b)` pair needs exactly one independent solve (symmetric `a=b` states need
one dealer; asymmetric pairs need both dealers, but one dealer of `(a,b)`
covers the mirror of `(b,a)`'s other dealer). This exactly matches the real
10×10 fleet (9 workers, dealer-0 only, `mv(10,10,1) = 1 − mv(10,10,0)`).
Counting ordered `(a,b)` pairs with `a,b ∈ [0,10]` by `min(a,b)` gives, per TC:
`{1,3}`=4, `{1,3,6}`=21, `{1,3,6,9}`=39, `{1,3,6,9,12}`=57 (sums to 121 = 11²,
the full non-mão grid). ×9 TCs: 36 / 189 / 351 / 513 jobs.

### Refined cost estimate (supersedes the old Fermi-based full-game range)

Real N2 Custom pricing pulled from the Cloud Billing Catalog API (Americas):
core $0.033192/$0.014310 on-demand/spot per vCPU-hr, RAM $0.004449/$0.001918
per GiB-hr (within the 8 GB/vCPU base ratio), extended RAM $0.009550/$0.004317
per GiB-hr (beyond it). The real fleet already runs on 2-vCPU custom-extended
machines (`n2-custom-2-76800-ext`) since **the solve is single-threaded — CPU
count only matters for the few-minute build step**, so cost scales almost
entirely with RAM, not vCPUs. RAM/job and time/job scaled linearly off the
10×10 anchor's real production numbers: 75 GB provisioned RAM (vs. ~57-61 GB
measured true peak), 9-TC fleet wall times 4.90-7.56h (mean 6.37h, 900-930
iters, ε≈2.5e-4).

| tier | jobs remaining | RAM/job | $/hr spot | time/job (floor) | tier cost (spot / on-demand) |
|---|---:|---:|---:|---:|---:|
| {1,3} Stage B (9×9 + 10×9, both dealers — 10×10 itself already solved) | 27 | 75 GB | $0.31 | 6.4h | $54 / $121 |
| {1,3,6} | 189 | 245 GB | $1.05 | 20.8h | $4,127 / $9,155 |
| {1,3,6,9} | 351 | 649 GB | $2.79 | 55.1h | $53,956 / $119,484 |
| {1,3,6,9,12} (full ladder) | 513 | **1.54 TB** | $6.65 | 131h (~5.5d) | $447,291 / $989,918 |
| **Total remaining** | **1,080** | | | | **≈$505K spot / $1.12M on-demand** |

This replaces the old, never-validated "$303k–$3.35M spot, full game" range
(`SOLVER_PLAN.md` §13) with a ~2.2x-wide band instead of ~10x-wide, and
identifies where the cost and risk actually concentrate: the
`{1,3,6,9,12}` tier is ~89% of the remaining spot cost, **and** its 1.54 TB/job
requirement likely exceeds what an N2 custom-extended VM can provision (N2's
largest predefined config tops out at 864 GB) — it would probably need GCP's
M2/M3 memory-optimized family (pricier per GB) or a disk-backed architecture
change, neither of which is priced in above. Treat that tier's number as an
optimistic floor.

**Known unquantified risk:** all time estimates assume every tier converges
in ~900 iterations, matching the 10×10 anchor. Deeper raise ladders are more
strategically complex and plausibly need *more* iterations to reach the same
ε, not just more per-iteration cost from a bigger tree — this session only
measured tree size, not convergence rate, for the three untested tiers.
Tree size was also only measured at TC0; the real 10×10 fleet's own wall-time
spread (4.90-7.56h across TCs at presumably similar iteration counts) implies
real per-TC variance this single-TC measurement doesn't capture.

---

## 2026-07-14 — Per-infoset BR-gap export pilot (11×10/tc0/dealer-1)

The planned full-tree quality artifact was run against the retained teacher2
checkpoint and tree cache. It contains **2,746,361** reachable info-set rows
of `(table_idx, br_value, eq_value, gap, weight)` at 20 bytes each, plus its
48-byte validated header: **54,927,268 bytes / 52 MiB** raw. Gzip reduces the
lazy Study payload to **13 MiB**. The shallow and depth-3–4 chart windows were
re-exported with additive `table_idx` row keys; every row in both windows has
a valid key. This is a storage/product measurement, not a new convergence
benchmark: the existing exact BR pass was reused, with no CFR iterations.

---

## 2026-03-23 — Initial Infrastructure Benchmark

**Setup:** First working version of CFR+ with pre-built game trees. No convergence testing — this validates infrastructure only.

### Configuration
- Score: 11×11 (mão de onze — no raises available)
- Turnup class: 0 (blocked_plain_level = 0)
- Iterations: 10 (infrastructure test only)
- Machine: local development machine
- Build: `cargo run --release`

### Results

| Metric | Value |
|--------|-------|
| Abstract deals enumerated | ~140,000 |
| Total game tree nodes (all deals × 2 dealers) | ~100M |
| Unique information sets | ~11.2M |
| Tree build time | ~115s |
| Per-iteration time | ~11s |
| Estimated memory (trees + strategy) | ~5.8 GB |

### Notes
- 11×11 has the **smallest** game trees of any score state because mão de onze disables raising. Lower scores will have larger trees.
- Memory is dominated by pre-built game trees (~5 GB), not strategy table (~800 MB).
- No convergence measured — this was purely to validate the infrastructure works.

---

## 2026-03-24 — CFR+ vs MCCFR Convergence Comparison (GCP)

**Setup:** Head-to-head comparison at equal wall time. Both algorithms given ~3300s on the same hardware.

### Configuration
- Score: 11×11, Turnup class: 0
- Machine: **GCP c2-standard-8** (8 vCPUs, 32 GB RAM, compute-optimized)
- Provisioning: on-demand (standard)
- Build: `cargo run --release --bin solve -- compare --iters 100`
- CPU: ~100% single-core (solver is single-threaded)
- Memory: ~12.8 GB RSS (~40% of 32 GB)

### CFR+ Results (100 iterations, 3299s)

| Iterations | Exploitability | Wall time (s) |
|-----------|---------------|--------------|
| 1 | 1.500118 | 240 |
| 5 | 0.649266 | 368 |
| 10 | 0.278750 | 546 |
| 20 | 0.119625 | 837 |
| 50 | 0.061883 | 1766 |
| 100 | 0.050954 | 3299 |

**Per-iteration breakdown:**
- Tree build: **218s** (one-time)
- Per CFR iteration: **~16s**
- Per exploitability computation: **~15s**
- Total per logged iteration: **~31s** (iteration + exploitability)

### MCCFR Results (8.35M iterations, 3305s)

| Iterations | Exploitability | Wall time (s) |
|-----------|---------------|--------------|
| 10,000 | 1.082 | 8 |
| 100,000 | 0.835 | 44 |
| 500,000 | 0.522 | 202 |
| 1,000,000 | 0.410 | 393 |
| 2,000,000 | 0.298 | 775 |
| 4,000,000 | 0.197 | 1551 |
| 6,000,000 | 0.158 | 2326 |
| 8,350,000 | 0.126 | 3305 |

**MCCFR observations:**
- Speed: ~2,530 iterations/second
- Exploitability measured every 10k iterations using 2000 deals (approximate)
- Convergence: much slower than CFR+ at equal wall time

### Head-to-Head at ~3300s Wall Time

| Algorithm | Final exploitability | Ratio |
|-----------|---------------------|-------|
| **CFR+** | **0.051** | **1.0×** |
| MCCFR | 0.126 | 2.5× worse |

**Verdict: CFR+ wins decisively** when the full tree fits in memory. MCCFR is not competitive at this tree size. MCCFR only makes sense when trees exceed available RAM.

### Convergence Rate Analysis

| Interval | Expl reduction | Iter ratio | Effective rate |
|----------|---------------|------------|---------------|
| iter 5→10 | 0.649→0.279 = 2.3× | 2× | ~O(1/T^1.2) |
| iter 10→20 | 0.279→0.120 = 2.3× | 2× | ~O(1/T^1.2) |
| iter 20→50 | 0.120→0.062 = 1.9× | 2.5× | ~O(1/T^0.7) |
| iter 50→100 | 0.062→0.051 = 1.2× | 2× | ~O(1/T^0.26) |

**Key insight:** Fast early convergence (~O(1/T)) that slows dramatically in the tail. This is typical of CFR+ — big mistakes are corrected quickly, but fine-tuning the mixed strategy grinds. The overall 100-iteration trend is roughly **O(1/T^0.74)**, better than the theoretical O(1/√T) worst case, but with diminishing returns at higher iterations.

### Projections to ε = 0.01

Extrapolating conservatively from the observed tail convergence:

| Target ε | Est. iterations | Wall time per TC* | BR win % |
|----------|----------------|-------------------|----------|
| 0.050 | 100 | 55 min | 52.5% |
| 0.030 | ~250 | 2.2 hr | 51.5% |
| **0.010** | **~700-900** | **6-8 hr** | **50.5%** |
| 0.003 | ~3,000 | 26 hr | 50.15% |

*Including exploitability computation every iteration. Computing every 10 iterations would roughly halve these times.

---

## 2026-03-25 — DCFR Follow-Up (11×11, TC 0)

**Setup:** Same top-level benchmark target as the CFR+ run, but using
`DCFR(alpha=1.5, beta=0, gamma=2)` with exploitability computed every 10
iterations.

### Configuration
- Score: 11×11, Turnup class: 0
- Algorithm: DCFR (`alpha=1.5`, `beta=0`, `gamma=2`)
- Max iterations: 120
- Exploitability cadence: every 10 iterations
- Log: `results/dcfr-120iters.log`

### Results

| Iterations | Exploitability | Total wall time (s) |
|-----------|---------------|---------------------|
| 1 | 1.500118 | 230.0 |
| 10 | 0.186249 | 392.1 |
| 20 | 0.075889 | 574.9 |
| 30 | 0.058882 | 756.7 |
| 50 | 0.052225 | 1123.8 |
| 80 | 0.049317 | 1673.3 |
| 100 | 0.048250 | 2037.0 |
| 120 | 0.047481 | 2396.4 |

### Takeaways

- DCFR is the current best measured algorithm on the `11×11`, `TC 0` benchmark.
- The early-iteration advantage over CFR+ is real, but the tail still slows
  substantially after ~50 iterations.
- `expl_every=10` is good enough for progress tracking and avoids paying the
  exploitability cost on every single iteration.

---

## 2026-06-28 — Full 11×11 Subgame Solve (all 9 turn-up classes, 3 VMs)

**Setup:** First solve of the **entire** 11×11 subgame — all 9 turn-up classes
TC 0–8 — to a lax exploitability target, using the new production
checkpoint/resume path (`solve solve-tc`, commit `00e8d3c`). Three
`e2-standard-8` GCE VMs (8 vCPU, 32 GB) ran in parallel, each handling 3 TCs
sequentially (peak RSS ≈ 16 GB/TC means only one TC fits in memory at a time):
`truco-solver-bench` → TC 0–2, `-bench-2` → TC 3–5, `-bench-3` → TC 6–8. The two
extra VMs were cloned from a machine image of the first (carries rust + repo +
built binary). Each VM self-shut-down after its last TC.

### Configuration
- Score: 11×11, all turn-up classes 0–8
- Algorithm: CFR+ (production `cfr.rs` path, not `cfr_experiment.rs`)
- Per-TC wall-clock budget: 3300s (`--time-budget 3300`)
- Exploitability cadence: every 5 iterations
- Checkpoint: full state every 1200s + final (`--checkpoint-every 1200`)
- Total run: ~2.9h wall (3 TCs × ~3360s per VM, in parallel)

### Results

| TC | Start expl | Final expl | Iters | Game value (P0) |
|----|-----------|-----------|-------|-----------------|
| 0  | 1.361071  | 0.053298  | 81    | 0.000078 |
| 1  | 1.361137  | 0.052511  | 83    | 0.000175 |
| 2  | 1.361264  | 0.051360  | 89    | 0.000095 |
| 3  | 1.361425  | 0.048649  | 107   | 0.000115 |
| 4  | 1.361582  | 0.050258  | 90    | 0.000092 |
| 5  | 1.361683  | 0.051160  | 90    | 0.000123 |
| 6  | 1.361669  | 0.050367  | 100   | 0.000072 |
| 7  | 1.361466  | 0.051971  | 90    | 0.000135 |
| 8  | 1.360992  | 0.051141  | 84    | 0.000183 |

Mean final exploitability ≈ **0.0511** (band [0.0486, 0.0533]).

### Sanity checks (zero-sum invariants — all pass)
- **Exploitability ≥ 0 and strictly monotonically decreasing**: 0 upticks across
  all 9 trajectories (1.361 → ~0.05).
- **Game value (P0) ≈ 0** for every TC (≤ 0.0002), i.e. match_value(11,11) ≈
  0.50005 ≈ 0.5 — exactly as symmetry demands. Strong correctness signal.
- **Cross-TC consistency**: all 9 final values within a 0.005-wide band; all
  start at ~1.361 (uniform-strategy exploitability is nearly TC-independent).
- All 9 runs `rc=0`; all 3 VMs cleanly self-shut-down.

### Takeaways
- The 9 turn-up classes are genuinely independent and parallelize perfectly: a
  3-VM split solved the whole subgame in ~3h wall instead of ~9h serial.
- CFR+ at 11×11 hits the same ~0.05 convergence wall after ~80–100 iters seen in
  the single-TC benchmarks; getting below ~0.05 will need the discounting work or
  far more iterations. **This is exactly what the new `--resume` is for**: each
  TC left a 1.7 GB full-state checkpoint (regrets + strategy + iteration) on its
  VM's (stopped, not deleted) disk, so a later run can `--resume` and keep
  improving from iteration ~85+ rather than restarting.
- Per-TC: 11.2M info sets, ~235s tree build, ~38s/CFR+ iter, 1.7 GB checkpoint +
  1.4 GB average-strategy file.

---

## 2026-06-30 — 11×10 and the mão-de-onze wall (11×11, TC 0-derived numbers)

Working toward 11×10 / 10×10. Key measured numbers and outcomes (full narrative in
`RESEARCH_NARRATIVE.md`):

### Pé (dealer) advantage at 11×11 (TC 0)
Computed from the solved 11×11 strategy via per-dealer game value (`solve
dealer-advantage`):
- dealer (pé) win% = **0.5564**, non-dealer = 0.4437 (average 0.5000, ✓ symmetry).
- So being the dealer at mão de onze is worth **+5.6 percentage points**. This
  invalidated the old dealer-*averaged* match-value table → made the solver
  dealer-exact (mv indexed by score×dealer; continuations use `mv(·, 1−dealer)`).

### 11×10 brute-CFR wall (the failed direct approach)
CFR+ on 11×10 (accept/fold *learned*), eps 0.05 target:

| iter | 5 | 25 | 45 | 85 | 105 |
|------|------|------|------|------|------|
| expl | 0.59 | 0.40 | 0.37 | 0.34 | 0.337 |

Flattens at ~0.33 (slower than 1/T). DCFR is identical (0.374 vs CFR+ 0.367 at
iter 45) → **structural, not algorithmic**. Not a bug (engine model verified).

### Freeze-accept decomposition (the working direction)
Freeze the accept per dealer, CFR-solve only the card play, iterate the accept set
on equity. Correctness check passed exactly: **round 0 (accept-all) per-dealer
value = 0.5564 / 0.4437 = the 11×11 numbers** (accept-all 11×10 *is* 11×11).
Accept set converges in ~1 round; decider value rises to ~0.634 both positions
(folding weak hands is worth ~13pp and ~equalises the pé gap).
- First cut plateaued at **exploitability ~0.26** (better than brute 0.33, not
  ~0.05). "Average inertia" hypothesis (warm regrets / reset average) **refuted**
  — no change (0.270 vs 0.257).
- Root cause hypothesis: folded hands get zero reach so their "value if accepted"
  was mis-measured (average play, not best-response). Fixed the equity to use
  **best-response** value — but it **did not** crack it.
- **What the per-player exploitability split showed** (BR_p0 = decider, BR_p1 =
  opponent; each summed over both dealer trees):
  - opponent best-response gain ≈ **0.025** → *card play is solved*;
  - decider best-response gain ≈ **0.51** → the whole residual is the *accept*;
  - accept set does not converge (403 → 301 → 349, growing).
- Conclusion: **freeze-and-iterate is fictitious-play on a screening game and does
  not converge the accept.** Indicated fix: add **position to the info set** (the
  accept node is currently dealer-shared → one position-averaged policy, part of
  why brute CFR walled) and let plain dealer-exact CFR learn the accept jointly;
  test vs the 0.33 wall. Fallback: fictitious-play averaging of the accept sets.

### Ops notes
us-central1-a hit a persistent 8-vCPU **capacity stockout** (bench + all machine
types failed to start) → moved diagnostics to a throwaway `e2-standard-8` in
us-east1-b. Per-11×10-round cost ~1800s cold (~40 iters); build ~225s / 100M nodes
/ ~16 GB (peak ~23 GB with warm-start source held). Gotcha: never `pkill` the
`sleep` of a `sleep N; shutdown` backstop — it runs the shutdown immediately.

---

## 2026-07-02 — Position in the info set: the 11×10 wall is structural key-sharing, removed

No heavy solves this entry (local only, deal-limited configs); the numbers are
smoke-scale but the comparison is exact A/B on identical configs.

### The reframing
The dealer-0 and dealer-1 games at any score are **fully independent games** —
they interact only through the (already-known) match-value table. The 11×10 wall
was never a screening-convergence problem for CFR: it was the implementation
tying two different games together through shared info-set keys, because
`InfoSet` did not encode position. Two kinds of cross-dealer-tree merging:

1. **The accept/fold node** (empty history): identical key in both trees ⇒ CFR
   could only learn one position-averaged accept policy, while the best response
   (computed per tree) accepts per position ⇒ a hard exploitability floor.
2. **Card-play aliasing** (subtler, newly identified): `ActionHistory` records
   *what* was played but not *who* acted, so the same visible string can mean
   "opponent led X, I played Y" in one dealer tree and "I led X, opponent played
   Y" in the other — same key, different states. This exists at 11×11 too and
   may be part of its ~0.05 floor.

Fix: `InfoSet.is_dealer` (position). Both merging classes disappear; the two
dealer games become disjoint in key space, so plain dealer-exact CFR converges
each as an ordinary EFG — the accept is learned jointly with the card play,
which is exactly what CFR's regret matching + averaging is for.

### A/B on an identical tiny 11×10 config
Config: TC 0, 300 strided deals (subsampling is now strided — a prefix gives
player 0 only ~1 distinct hand and shows no wall at all), CFR+,
mv(11,11,·) seeded 0.5564/0.4437, exploitability every iteration.

| iter | pre-fix (shared keys) | post-fix (position in key) |
|------|----------------------|----------------------------|
| 26   | 0.1254 | 0.0408 |
| 51   | 0.1036 | 0.0102 |
| 151  | 0.0971 | 0.0012 |
| 300  | **0.0965 (walled, ~flat)** | **0.0003 (still ~1/T)** |

Same wall shape as the full-scale brute-CFR 0.33 wall; the fix removes it
completely. Also verified: a `--dealer 0` filtered solve reproduces the joint
solve's dealer-0 game value to <1e-12 (disjointness ⇒ identical update
sequence), and every decider hand gets an accept node in *both* positions.

### Real-run design for 11×10 (capacity-flexible)
1. `solve set-mv` — seed `mv(11,11,0)=0.5564`, `mv(11,11,1)=0.4437` into
   `match_values.bin` (11×10's fold continuation reads these; defaults of 0.5
   are wrong for it).
2. Per TC, run **two independent jobs**: `solve-tc --score 11x10 --tc N
   --dealer 0` and `--dealer 1`, each with `--match-values`. Each builds ~half
   the joint tree (~50M nodes, roughly half the ~16 GB), so smaller/other-zone
   machines work — no dependency on the stranded us-central1-a capacity.
   Artifacts get a `.dN` infix (`tcN.d0.ckpt.bin`, `tcN.d0.bin`, `tcN.d0.gv`).
   **Target scale:** the historical joint exploitability sums both dealer
   games, so a single-dealer job should use `--eps 0.025` to match the joint
   ~0.05 bar (per-dealer numbers are on the half scale).
3. Optional: `--warmstart-from` the legacy 11×11 checkpoints — the loader
   detects the pre-position format and expands each entry into both position
   variants automatically.
4. Expectation: convergence like 11×11's card play (the opponent's
   best-response gain was already ≈0.025 under freeze-accept). If a ~0.05-ish
   floor persists, the next suspect is within-tree abstraction (not the accept),
   since the tiny config shows no residual accept coupling.
5. `mv(11,10,d)` then comes from the per-dealer `.gv` sidecars (TC-weighted),
   and 10×11 by symmetry; 10×10 unblocks after that.

### Addendum: aliasing census (same day)
Measured (build-only, strided deal subsets, TC 0) how many info sets differ
ONLY by position — i.e. would have collided across dealer trees pre-fix:

| deals | 11×11 aliased | 11×10 aliased |
|-------|---------------|----------------|
| 300   | 0 | 300 |
| 1,000 | 0 | 403 |
| 3,000–30,000 | 0 | **403 (saturated)** |

Two conclusions, one of which corrects a suspicion recorded above:

1. **11×10's aliasing is exactly the accept nodes** — 403 = the number of
   distinct decider hands (the same 403 from the freeze-accept oscillation).
   Nothing else was merged, which is also why the freeze-accept decomposition
   found the card play clean.
2. **11×11 has NO aliasing at all** (zero across a 100× density range; the
   history encoding turns out to disambiguate attribution — own face-down plays
   record the card while the opponent's record `OpponentPlayedHidden`, and
   face-up trick winners are computable from the string). So the ~0.05 at 11×11
   was *not* an abstraction floor — it is simply where the runs met their lax
   target — and the existing 11×11 solve, its per-dealer values 0.5564/0.4437,
   and its checkpoints are all exactly as good as their ~0.05 tolerance. **No
   11×11 re-solve is required before 11×10.** (Corrects the "may be part of the
   11×11 floor" suspicion above; open question 1c is answered.)

Since 11×11 had no truly-shared entries, the legacy loader's expand-to-both-
positions is *exact* for it: the spurious twin of each entry matches nothing in
a new build and is dropped on resume overlay. Caveat down the lattice: at
raise-enabled states (10×10 and below), `Raise`/`AcceptRaise` tokens are
attribution-ambiguous the same way the accept node was, so genuine card-play
aliasing may reappear there — position in the key inoculates those states too.

### Ops notes
- Joint (both-dealer) builds still work (`--dealer` omitted) and cost extra
  info-set memory only where entries were genuinely shared (at 11×10: the 403
  accept nodes — negligible).
- Old checkpoints/strategies load via a legacy-format fallback; new files are
  not readable by old binaries.

---

## 2026-07-02 (later) — Exact best response: all prior exploitability numbers were clairvoyant upper bounds

**Read this before comparing any exploitability number across entries.** The
historical `best_response_value` maximized independently inside every deal's
tree (`Σ_deals max` instead of the legal `max_per_info_set Σ_deals`), i.e. the
"best responder" could condition on the opponent's hidden hand. That
clairvoyant bound does not vanish at equilibrium and grows with info-set
pooling (deal density). Every exploitability figure in entries above is such a
bound; game values (avg-vs-avg) were computed by a separate, correct path and
are unaffected.

### How it was isolated (all local, deal-limited, 11×10 TC 0, CFR+ unless noted)
| probe | clairvoyant expl | exact expl |
|-------|------------------|------------|
| 300 strided deals, 300 iters | 0.0003 | ≈ same (no pooling → measures agree) |
| 1,000 deals, 300 iters | 0.1596 | **0.0093** (@290) |
| 1,000 deals, 3,000 iters | 0.1533 (hard floor) | — |
| 3,000 deals, 300 iters | 0.2569 | **0.0118** |
| 3,000 deals, DCFR / SyncCFR+ / PCFR+ | 0.2557 / 0.1546* / 0.1547* | — |

*SyncCFR+/PCFR+ at 1,000 deals. Four algorithm variants on one identical
floor ⇒ the metric, not the dynamics. The 10× iteration probe (0.1596→0.1533)
ruled out slow convergence: even worst-case 1/√T predicts ~0.05, and a true
floor contradicts CFR's guarantee on a finite zero-sum game.

### Reinterpretation of earlier entries
- 11×11 "walls" at CFR+ 0.0510 / DCFR 0.0475: the clairvoyance gap of
  essentially converged strategies. True exploitability unknown; measurable
  now with `solve eval-ckpt --ckpt <11x11 ckpt>`.
- 11×10 walls 0.33 (brute) / 0.26 (freeze-accept) / ~0.15/game
  (post-position-fix fleet): all clairvoyant. The position fix itself remains
  validated (tiny-config A/B where the measures agree; aliasing census).
- The freeze-accept "decider BR gain ≈ 0.51" diagnostic was clairvoyant too —
  the accept-set oscillation story stands, but its magnitude was inflated.

### New tooling
- `best_response_value` is now the exact per-info-set best response
  (counterfactual weights top-down; per-info-set argmax deepest-first; resolved
  profile evaluated top-down). Clairvoyant version kept as
  `best_response_value_clairvoyant` (diagnostics; solve-asym path).
- `solve eval-ckpt --ckpt PATH [--max-deals N] [--dealer D]` — exact
  exploitability + per-dealer game values of any checkpoint.
- `CfrAlgorithm::{SyncCfrPlus, PcfrPlus}` — synchronous sweeps and predictive
  regret matching, kept as first-class variants (built to falsify the dynamics
  hypotheses; may still earn their keep at lower-score states).

### Fleet v2 (in flight at time of writing)
The 18-job 11×10 fleet was relaunched on the fixed binary, resuming from the
GCS checkpoints (~iteration 1,200-1,500 per job, CFR+), now with `--eps 0.01`
per dealer game under the exact measure. Expectation based on the mid-density
probes: the resumed strategies are already at or near the target and jobs
should finish within a handful of exploitability checks.

---

## 2026-07-02/03 — 11×10 SOLVED (all 9 TCs × both dealers, exact measure)

First asymmetric mão-de-onze state solved end-to-end. Fleet: 2× n2-highmem-16
(us-east1-c), one VM per dealer, 9 concurrent single-threaded jobs each, CFR+,
`--dealer`-filtered builds (~half memory), full campaign (both fleet
generations plus probes) ≈ $15.

### Final exact exploitability (per dealer game; ±1 scale, halve for win-prob pp)
| TC | dealer 0 | dealer 1 |
|----|----------|----------|
| 0–8 range | 0.0115–0.0121 (max-iters 2000 cap) | 0.0058–0.0065 (eps target hit) |

Dealer-0 games stopped at the 2,000-iteration cap just above the 0.01 target
(≈1/T tail, extrapolated exactly there); resumable from checkpoints if a
tighter target is ever needed. All artifacts (strategies, full-state
checkpoints, `.gv` sidecars, convergence logs) under
`gs://truco-solver-runs/11x10-20260702/`.

### Match values (TC-weighted; P0 = the player at 11)
- **mv(11,10, dealer=0) = 0.6218** (decider deals / is pé)
- **mv(11,10, dealer=1) = 0.6297** (decider out of position)
- mirrors: mv(10,11,0) = 0.3703, mv(10,11,1) = 0.3782
- Per-TC spread within a dealer ≤ 0.002 — turn-up class barely matters, again.
- Canonical seeded table: `gs://truco-solver-runs/11x10-20260702/match_values.bin`
  (also includes the 11×11 constants).

**The position inversion.** At mão de onze the pé advantage (+5.6pp at 11×11)
not only shrinks — it flips sign: the decider is *better off out of position*
(+0.8pp). Mechanism: folding concedes one point and flips the dealer for the
11×11 continuation, so the out-of-position decider's fold option rotates them
*into* position (fold value 0.5564) while the in-position decider's fold
rotates them out (0.4437). The accept/fold option is worth ~13pp overall
(0.556/0.444 → 0.622/0.630) and its value is asymmetric in exactly the
opposite direction of the positional edge.

Tolerances: these values inherit (a) per-subgame exact exploitability above
and (b) the 11×11 constants' (unmeasured-but-likely-small) true gap; the
anytime-refinement loop (resume checkpoints at lower eps, cascade re-solves)
tightens both later. 10×10 is now unblocked — its only non-terminal
continuations are stake-1 outcomes into 11×10/10×11 (truco already decides the
match at 10×10), and by symmetry only the dealer-0 game needs solving:
mv(10,10,1) = 1 − mv(10,10,0).

---

## 2026-07-03 — Synchronous sweeps: ~10× fewer iterations (exact measure)

The historical iteration recomputed strategies from live regrets mid-sweep
(asynchronous Gauss-Seidel). Freezing the strategy for the whole deal sweep
(`--algo sync`, buffered instantaneous regrets folded at iteration end — the
textbook CFR+ iteration) converges dramatically faster under the exact
measure, replicated at two densities (11×10 TC0):

| iter | async 3000d | sync 3000d | async 1000d | sync 1000d |
|------|------------|-----------|------------|-----------|
| 100  | 0.0215     | **0.0023** | ~0.017     | ~0.002    |
| 400  | 0.0103     | **0.00017**| 0.0080(290)| **0.00017** |

Sync's tail decays ≈T^-1.9 (vs async's ≈T^-0.6 — which retroactively explains
the slow 10×10 tail). Consequences:
- ~10× fewer iterations to any target ⇒ ~10× cheaper solves immediately
  (`--algo sync` now used by all fleet scripts; full-scale pilot on 10×10 TC7
  in flight).
- Sync sweeps are read-only on regrets during traversal ⇒ deals can be fanned
  across threads (per-thread or atomic pending buffers), multiplying cores per
  job and dividing RAM-hours — the next optimization target.
- Revised full-game Fermi: ~$115k (async, single-thread) → **~$10-12k** on
  iteration count alone; node compaction + threaded sweeps target another
  ~4-8× (→ ~$2-4k).

---

## 2026-07-03 (later) — Full-scale sync validation + the optimization stack

### tc7 pilot: sync at full 10×10 scale
Fresh SyncCFR+ run on the real 10×10 TC0 dealer-0 game (352M nodes):
**0.0095 exact exploitability at iteration 90** (~2h wall on one core incl.
build). The async fleet had needed ~1,000+ iterations to reach only ~0.011
before its budget. Full-scale confirmation of the ~10× iteration reduction.

### 11×9 first datapoint
The single launched 11×9 worker (async, 2-vCPU/20 GB, $0.176/h) converged to
0.00996 at iteration 740 in under an hour: **≈ $0.18 for a solved subgame** of
the mão-de-onze class. The remaining 17 jobs are deferred (cost-optimization
pause) and will be cheaper still under sync.

### Shipped optimization stack (all lossless; plan 70)
| Lever | Mechanism | Measured effect |
|---|---|---|
| Sync sweeps (`--algo sync`) | textbook frozen-strategy iteration | ~10× fewer iterations (tiny/mid/full scale) |
| Dense accumulators | `table_idx` indexing, no per-visit hashing | 4.9× faster iterations |
| Packed tree arena | 12 B nodes + shared edge array, i8 payoffs | ~4.5× less tree RAM, ~10% iter cost |
| Parallel sync sweeps (`--jobs N`) | rayon over deals, atomic accumulation | equivalence proven; scaling measured on the bench VM (below) |

10×10 tc1–6 (async, mid-flight) were deliberately killed as economically
obsolete — re-solving each with the new stack costs well under $1 vs 7–9 more
async hours each. Checkpoints retained.

### Bench VM matrix (n2-standard-16, 10×10 TC0 d0) — MEASURED, with a correction

**Correction to an earlier claim in this file:** the "~11 GB RSS / 60→11 GB
packed-trees win" cited above was the solver's *internal estimate*
(`estimate_memory_bytes`), NOT measured RSS — a bench bug left `rss_kb=` empty
and the estimate slipped through as if measured. Two `ps`-measured bench runs
now show **actual peak RSS ≈ 57–61 GB** at 10×10 TC0 dealer-0. Packed trees did
shrink the *tree* ~4.5×, but the tree is only ~15% of the footprint (the rest:
the 39.5M-entry dense accumulator table of per-info-set heap vectors, the
info-set metadata held in *both* a Vec and a HashMap, and the build-time
registry HashMap), so overall peak barely moved. `estimate_memory_bytes`
undercounts by ~5× and should not be trusted for machine sizing.

Measured (persistent-buffer sync, exploitability suppressed):
| jobs | peak RSS | s/iter (incl. 2 expl passes / 12 iters) |
|------|----------|------------------------------------------|
| 1 | 57.7 GB | 77.7 |
| 4 | 60.4 GB | 66.8 |
| 8 | 61.2 GB | — |

**Within-job threading is a measured dead end (~1.26× on 4 cores after backing
out the serial exploitability pass).** The sweep is memory-latency-bound:
random pointer-chasing into a ~15 GB `Vec<InfoSetData>` of 39.5M little heap
vectors — threading only buys a little memory-level parallelism. Two rewrites
(shared atomics → false sharing; naive thread-local → per-iteration alloc
storm, RSS 11→55 GB, 95% system CPU; then persistent buffers → correct memory
but still ~1.26×) confirm it. Conclusion: **rely on cross-job parallelism**
(one core per subgame, perfect scaling) and treat the ~57 GB footprint as the
real cost driver. `--jobs` is retained (correct, equivalence-tested) but off by
default.

### Converged $/job (measured)
10×10 TC0 d0 to 0.0098: 90 iters, 4352 s solve on n2-standard-16 = **~$0.98/job**
at that (over-provisioned, 16-vCPU) machine. Single-core on a right-sized
~64 GB box (~$0.30–0.50/hr) ≈ **$0.5–0.7/job**; spot ≈ $0.15–0.25. The 64 GB
requirement (not 16 GB) is the correction that matters for fleet sizing.

### Revised full-game Fermi (honest)
Sync's ~10× iteration cut stands (the load-bearing win). Packed trees help less
than claimed in absolute RSS. Threading doesn't help. With ~57 GB/job scaling
up with node count for deeper states (a min≤2 state could need ~150–190 GB):
**~$3–6k**, still dominated by unmeasured deep-state tree sizes. Open levers:
(1) flatten the dense table to contiguous SoA + drop duplicated metadata
(target ~57→~25–30 GB, also cuts the pointer-chase latency); (2) MCCFR for the
deep band (no prebuilt trees → breaks RAM∝nodes). Band-size probes remain the
dominant Fermi uncertainty.

---

## 2026-07-04 — Engine review: allocation-free `Engine::clone` (tree builds 1.6× faster)

The full-engine review (validation hardening + rules audit) included one
solver-relevant change: `Card.id` became `Arc<str>` and hands/rounds moved to
inline `SmallVec` storage, so the per-tree-node `Engine::clone` in the
builders is now a memcpy plus refcount bumps instead of ~15–20 heap
allocations.

Local perf probe (M-series laptop, 3000 strided deals, 11×10 TC0):

| metric | before | after |
|---|---|---|
| tree build (3000 deals) | 1.44 s | **0.89 s** (1.62×) |
| solve s/iter | 0.021 | 0.021 (unchanged — CFR never touches `Engine`) |

Build-time-only win, but it applies to every band build (~90 for the full
game) and to any treepack-cache miss. Engine behavior is bit-identical;
the whole fixture corpus (now 67 fixtures), the TS-mirror corpus, and a new
200-match randomized playout/round-trip test all pass. Wire formats (JSON)
unchanged; no solver artifact impact.

## What Changed After The First Benchmark

Some items that were previously just "next optimizations" now exist in the
codebase:

- `solve_until` supports configurable exploitability cadence via `expl_every`.
- DCFR is implemented alongside CFR+ and can be selected from the solver CLI.
- The top-down pipeline parallelizes across independent score/turnup jobs and
  checkpoints `match_values.bin`.

The remaining unknowns are increasingly about lower-score tree sizes, full-grid
operations, and saved-strategy inspection rather than basic solver plumbing.

---

## 2026-07-05 — Teacher stage A: the full 11-column solved at ε=1e-5

The whole `11×{0..11}` column solved to teacher grade in one campaign
(`teacher2` + `teacher2b` resume). `DONE` at
`gs://truco-solver-runs/teacher2-20260704/DONE` (2026-07-05T15:49Z).

### Configuration
- **216/216 gv sidecars** = 12 rungs (11×11…11×0) × 9 TCs × 2 dealers, each
  subgame solved to exploitability **ε = 1e-5** (sync CFR+, `solve-tc --algo
  sync`, `--expl-every 20`).
- VM: `n2-highmem-16`, us-central1-a. Band treepacks (mão-p0 per tc/dealer)
  built once and reused across rungs. mv table propagated rung-by-rung
  descending; `set-mv` after each rung, mirror `mv(s,11)=1−mv(11,s)`.
- Anchor cost (from teacher1, same grade): 11×11 tc0 = 1380 iters / 78 min /
  5.61M info sets. `export-teacher` σ̄-vs-σ̄ pass over that = **0.6 s** for 5.6M
  info sets; game value reproduced to |Δ| = 7.7e-14 vs the solver.

### Results — tc0/d0 game values (P0, ±1 match space)
Clean monotone descent as the opponent's score falls (mão's edge grows):

| score | v(P0) | mão match-win |
|------:|------:|--------------:|
| 11×11 | +0.1080 | 55.4% |
| 11×10 | +0.2683 | 63.4% |
| 11×9  | +0.4406 | 72.0% |
| 11×8  | +0.5905 | 79.5% |
| 11×7  | +0.7286 | 86.4% |
| 11×6  | +0.7831 | 89.2% |
| 11×5  | +0.8527 | 92.6% |
| 11×4  | +0.8861 | 94.3% |
| 11×3  | +0.9224 | 96.1% |
| 11×2  | +0.9395 | 97.0% |
| 11×1  | +0.9590 | 98.0% |
| 11×0  | +0.9680 | 98.4% |

### Stage-B pricing anchor (10×10 scout, same campaign)
- 10×10, ε=**2.52e-4**: 900 iters, **7.55 h/job**, 39.5M info sets, 11.25 GB.
  This is the cost anchor for planning Stage B ({1,3} triangle) — an order of
  magnitude larger than an 11-column rung.

### Ops notes / failures
- **teacher2 v1** wrote a false `DONE` (git archive lacked `--prefix=truco/` →
  `cd truco` failed → jobs no-oped). Fixed: `--prefix=truco/` + a `fail()` guard
  that writes `FAILED` (never `DONE`), uploads the log, and powers off.
- **teacher2 v2** filled its 300 GB disk at the 11×3 rung (8 rungs × 18 jobs of
  checkpoints+strategies). Resumed as **teacher2b**: restore mv table, solve
  11×3…11×0, `rm -f solutions/11xS/*.bin` after each rung's GCS upload (keep gv
  + mv table). Recipe: `tools/gcp/startup-teacher2b-resume.sh`.
- Bucket now 363 GiB; `.ckpt.bin` checkpoints (~half) are prunable dead weight.

### Downstream
- Final `match_values.bin` copied into the repo. `truco-frontend/public/study/`
  charts regenerated off this column (Study lab). See **plan 74** for the
  dominated-action-residue finding these clean solves exposed and the
  purify-then-certify plan built on top of them.

---

## 2026-07-09 — Study export purification/certification for tc0/d0

Implemented plan 74 for the current `11×{0..11}`, tc0/dealer-0 Study column.
The source `.teach` tensors remain raw. The chart export now writes purified
`p`, raw average-strategy `raw_p`, unchanged `q`, and a
`study-purification-certificate/v1` block. Certification uses the exact legal
per-info-set BR pass against both raw and purified profiles.

### Step-0 residue sweep

Command: `solve audit-teach-residue --teach-dir <scratch>/teacher-refresh/teach`.
The histogram is intentionally capped to individually-small actions
(`p < 1%`), matching the purifier. An uncapped all-action sweep counts
strategic/non-residue rows and is not the product diagnostic.

| Q-gap | touched info sets | max mass/info set | info sets ≥0.1% | ≥0.5% | ≥1% | ≥3% |
|------:|------------------:|------------------:|----------------:|------:|----:|----:|
| >1pp  | 3,640,606 / 67,337,909 | 2.9961% | 574,535 | 233,778 | 11,551 | 0 |
| >5pp  | 3,272,444 / 67,337,909 | 2.9948% | 515,808 | 209,022 | 9,322 | 0 |
| >20pp | 2,040,962 / 67,337,909 | 2.9948% | 314,809 | 125,224 | 5,346 | 0 |

Chosen thresholds from the sweep:
- Export assertion: for actions with `p < 1%`, fail if any info set carries
  more than **3%** total mass at Q-gap **>5pp**.
- Purification: zero actions with `p < 1%` and Q-gap `>5pp`, then renormalize.
- Frontend display gate: suppress only when `p < 3%` and Q-gap `>1pp`; the
  "Raw residue" toggle uses `raw_p` and bypasses the gate.

### Certification results

Shallow and deep files for the same rung share the same certificate. Every
purified strategy certified at or below raw exploitability.

| score | raw ε | purified ε | actions zeroed | max mass removed |
|------:|------:|-----------:|---------------:|-----------------:|
| 11×11 | 9.9918e-6 | 5.2982e-6 | 326,507 | 2.983% |
| 11×10 | 9.9718e-6 | 3.2251e-6 | 333,642 | 2.995% |
| 11×9  | 9.9515e-6 | 4.1534e-6 | 292,166 | 2.932% |
| 11×8  | 9.5837e-6 | 5.0979e-6 | 333,066 | 2.990% |
| 11×7  | 9.9388e-6 | 5.7358e-6 | 330,754 | 2.964% |
| 11×6  | 9.6852e-6 | 5.5913e-6 | 323,597 | 2.985% |
| 11×5  | 9.2674e-6 | 6.3221e-6 | 321,922 | 2.925% |
| 11×4  | 9.9033e-6 | 7.2712e-6 | 321,929 | 2.964% |
| 11×3  | 9.4737e-6 | 8.5869e-6 | 314,731 | 2.960% |
| 11×2  | 9.1304e-6 | 8.7307e-6 | 307,129 | 2.977% |
| 11×1  | 9.9420e-6 | 9.7352e-6 | 298,920 | 2.940% |
| 11×0  | 8.9237e-6 | 8.9237e-6 | 0 | 0.000% |

### GCS Study mirror

On 2026-07-09 the checked-in purified/certified Study artifacts were mirrored to
`gs://truco-solver-runs/teacher2-20260704/study/` (25 objects: manifest + 12
shallow JSON + 12 deep JSON.GZ files, 60.5 MiB). This is a derived Study-chart
prefix: the raw teacher artifacts under `solutions/` and the local/GCS `.teach`
sources remain raw by design.

### Dormancy cleanup

The project is going dormant for a few days to a couple of weeks, so the stopped
Compute Engine fleet was deleted after confirming the useful artifacts are in
GCS. Deleted instances: `truco-solver-bench`, `truco-solver-bench-2`,
`truco-solver-bench-3`, `truco-teacher2`, `truco-11x9-tc0-d0`,
`truco-10x10-scout`, `truco-10x10-tc7`, `truco-11x10-d0`, `truco-11x10-d1`,
`truco-conductor`, `truco-membench`, and `truco-teacher1`. All had auto-delete
boot disks, so the 990 GB of stopped-VM Persistent Disk storage was removed too.
Post-cleanup checks: zero Compute Engine instances, zero persistent disks, zero
reserved static IPs. GCS run prefixes remain intact, including `teacher2-20260704`
(363.04 GiB), its purified Study mirror (60.52 MiB), `11x10-20260702` (28.86
GiB), `10x10-20260703` (69.85 GiB), and `11x9-20260703` (1.56 GiB).

---

## 2026-07-09 (later) — 10x10 micro-canary, resume certification, full 10x10 spot fleet

### Micro-canary (`truco-stageb-10x10-micro-tc8`, bounded 30-iter probe)
Validated the Stage-B recipe end to end before spending real compute: code
archive fetch/build, `match_values.bin` restore, periodic GCS rsync, checkpoint
save, DONE marker, clean `poweroff` — all green, no code changes needed. Tree
(352,297,258 nodes / 39,508,752 info sets) and the solver's memory-estimate
line (11,251 MB) matched the 2026-07-05 "10×10 scout" pricing anchor almost
exactly, on `n2-custom-4-76800-ext` (4 vCPU / 75 GB). Reminder: that
memory-estimate line is `estimate_memory_bytes`, the same ~5× undercount noted
2026-07-03 — true peak RSS is the ~57–61 GB measured elsewhere for this
subgame, not the printed 11.25 GB.

### Resume/checkpoint certification (`startup-resume-test.sh`)
Before committing to spot instances for the full fleet, tested the actual
preemption-recovery path rather than assuming it: launched a real 10×10 tc0/d0
job with `checkpoint_every=30s`, waited for the first checkpoint to land in
GCS (iter 1, expl 0.532901), `gcloud instances stop`, then `start`. Confirmed:
- checkpoint writes are atomic (`storage.rs::save_checkpoint_iter` writes to
  `<path>.tmp` then renames) — a kill mid-write cannot corrupt a checkpoint.
- reload correctly restored all 39,508,752 info sets and resumed iteration
  counting at 1 (`resumed from iteration 1 (39508752 info sets)`), not a
  restart from 0.
- post-resume iter-10 exploitability (0.188787) matched the original
  uninterrupted tc8 canary's iter-10 value (0.188127) well within the
  documented ≤0.002 per-TC spread — the resumed trajectory is the same
  computation, not a divergent one.
- caveat carried forward: each restart re-pays tree build (~5 min) and a full
  `cargo build --release` (the startup script `rm -rf`s the workspace on every
  boot), since only the GCS-synced checkpoint survives a preemption, not local
  disk state. Budget ~7–10 min of "redo setup" per preemption in ETA math.

Isolated GCS prefix (`resume-test-20260709`) and the test VM were deleted after
certification; nothing from this test is in the real fleet's artifacts.

### Full 10x10 fleet launched (`10x10-full-20260709`, target ε=2.5e-4)
9 SPOT workers, one per turn-up class (TC0–8), dealer 0 only (symmetry:
`mv(10,10,1) = 1 − mv(10,10,0)`), `n2-custom-2-76800-ext` (2 vCPU / 75 GB — CPU
count doesn't matter for the single-threaded solve, only for the few-minute
build step). `--instance-termination-action=DELETE` on preemption; a
`e2-small` watcher (`truco-10x10-watcher`) polls every 3 min and relaunches any
TC that's neither DONE nor currently alive, so preemptions self-heal without a
human. Launched ~2026-07-09T21:20Z; ETA ~7.5–8h to ε=2.5e-4 per job (the
2026-07-05 900-iter/7.55h anchor), so same-day completion expected.

**Quota gotcha (new):** `SSD_TOTAL_GB` is a separate 500 GB/region quota from
the more generous `DISKS_TOTAL_GB`, and **`pd-balanced` disks count against it
too**, not just `pd-ssd`. The 9th worker (tc8) failed to launch in us-east1
because a pre-existing, unrelated 50 GB disk (`truco-export-10x10-scratch`)
plus the watcher's own 20 GB `pd-balanced` boot disk had already eaten 70 GB of
the region's 500 GB SSD headroom, on top of 4×100 GB workers there. Fixed by
routing tc8 to `us-west1` (untouched, 0/500 GB) rather than trimming disk
sizes; pushed the fix to the already-running watcher via
`instances add-metadata` + stop/start (the exact resume-tested restart path,
reused here for a config change instead of a preemption). Final split: 4
workers us-east1 (470/500 GB), 4 us-central1 (400/500 GB), 1 us-west1
(100/500 GB) — worth checking `SSD_TOTAL_GB` usage, not just `CPUS_ALL_REGIONS`
and `IN_USE_ADDRESSES`, before sizing the next multi-region fleet.

### Full 10x10 fleet — RESULTS (all 9 TCs, dealer 0, converged below ε=2.5e-4)

All 9 workers finished cleanly overnight, zero preemptions for the entire run
(watcher log shows nothing but "alive (RUNNING)" from launch to DONE — the
self-healing path was never actually exercised in production, only in the
`startup-resume-test.sh` certification). Watcher detected all 9 DONE markers
and self-terminated at 2026-07-10T04:57:16Z.

| TC | iters | final ε | wall time | v_d0 (P0, per-dealer) |
|---:|------:|--------:|----------:|-----------------------:|
| 0 | 910 | 0.000248 | 6.89h | +0.055711 |
| 1 | 920 | 0.000248 | 6.06h | +0.055725 |
| 2 | 900 | 0.000249 | 7.08h | +0.055824 |
| 3 | 920 | 0.000247 | 6.00h | +0.055665 |
| 4 | 930 | 0.000246 | 7.56h | +0.055869 |
| 5 | 920 | 0.000248 | 6.18h | +0.055837 |
| 6 | 920 | 0.000246 | 7.41h | +0.055947 |
| 7 | 920 | 0.000248 | 4.90h | +0.056228 |
| 8 | 900 | 0.000248 | 5.26h | +0.056148 |

All 9 land within a 0.00056-wide band (+0.055665 to +0.056228) — the same
"turn-up class barely matters" pattern already seen at 11×10, and a strong
cross-TC consistency check on correctness. Simple average v_d0 ≈ +0.0559; a
proper TC-weighted `mv(10,10,0)` (deal-frequency-weighted, mirroring
`aggregate_11x10.py`) is the natural next step if this feeds Stage B, but
hasn't been run yet — the per-TC spread is tight enough that it won't move
the headline number much.

**Iteration count landed above the 900-iter anchor** (900–930 vs. the
2026-07-05 scout's 900) — consistent, not a regression; the anchor was one
data point, this is nine independent confirmations at essentially the same
count.

**Wall-clock split by zone, not by TC**: us-east1 workers (tc0,2,4,6) all ran
6.9–7.6h; us-central1/us-west1 workers (tc1,3,5,7,8) all ran 4.9–6.2h despite
identical tree size (39,508,752 info sets, bit-for-bit the same across every
TC). Likely host-level noisy-neighbor variance on the us-east1 allocation,
not an algorithmic difference — worth a note for future multi-region timing
estimates, but not investigated further since it didn't block anything.

**Actual cost: ~$20** (vs. the ~$22–28 pre-launch estimate) — 58.3 worker-VM-hours
at the real SKU spot rate ($0.314/vCPU+RAM-hr for `n2-custom-2-76800-ext`,
pulled from the Cloud Billing Catalog API) ≈ $18.31, + ~$1.36 in `pd-ssd`
disk-hours, + ~$0.40 for the `e2-small` watcher's own on-demand runtime. Came
in under estimate mainly because zero preemptions occurred and average
per-job wall time (6.48h) beat the flat 7.65h planning assumption.

All artifacts (`.bin` strategies, full-state `.ckpt.bin` checkpoints, `.gv`
game-value sidecars, logs) are under
`gs://truco-solver-runs/10x10-full-20260709/d0/`. All 10 instances (9 workers
+ watcher) are `TERMINATED` but not deleted — their boot disks are still
billing until an explicit `instances delete` (`autoDelete` only fires on
delete, not on in-guest `poweroff`).

---

## 2026-07-11 — Epsilon-tremble warm-started refinement (all 5 shipped tc0 spots)

Implements QUESTIONS.md Q3 route (1). Warm-started `--resume` on the existing
teacher2-20260704 / 10x10-full-20260709 checkpoints with a new perturbed-game
tremble mode (`cfr::TrembleSchedule`, `--tremble-eps 0.05 --tremble-eps-end
0.01`, annealed) — every info set's strategy floored at `ε/|A| +
(1-ε)·σ(a)` for reach/regret/averaging, so previously near-zero-reach
branches finally get real training instead of uniform-noise regret matching.
Full mechanism writeup in `RESEARCH_NARRATIVE.md` 2026-07-11; outcome vs the
Q3 acceptance criteria in `QUESTIONS.md` Q3.

### Ops

Two short-lived on-demand GCE VMs (`us-east1-b`): `n2-highmem-8` for the four
11-column jobs (11×11 d0/d1, 11×10 d0/d1 — small trees, run in parallel as 4
background processes on one VM) and `n2-custom-2-76800-ext` (75 GB) for
10×10 d0 alone (needs the same ~57-61 GB peak RSS class as a fresh solve).
`--extra-iters N` (new `solve-tc` flag) resolves `max_iters = checkpoint_
iteration + N` once the checkpoint loads, so a resume-relative iteration
budget doesn't need a throwaway discovery pass. Four bugs surfaced and were
fixed along the way, all in the ops scripting, none in the solver: (1)
`$HOME` unset under GCE's metadata-script-runner combined with `set -u`
killed the script immediately after `rustup install`; (2) the debian-12 base
image ships no C toolchain (`apt-get install build-essential pkg-config`
before `cargo build`); (3) a bare `wait` after `exec > >(tee ...)` also waits
on the `tee` process spawned by the redirect, which never exits — the 4
11-column jobs finished correctly but the script hung after, idling
(harmless, just wasted ~20 min of billing before caught and stopped
manually — fix for next time: `wait $(jobs -p)` with explicit PIDs, not bare
`wait`); (4) the `export-chart` deep pass used `--max-depth 3` uniformly,
but 11×10's raise ladder reaches one level deeper than 11×11's mão-only
tree — the shipped 11×10 deep files are `--min-depth 3 --max-depth 4`, so
the first re-export undercounted rows 10× (13,214 vs 184,726) until this was
caught by the row-count verification step and fixed with a standalone
re-export against the already-computed `teacher.teach` (no re-solve needed).

### Iteration cost and budget

Measured cost with trembling ON confirms the mechanism's own documented
tradeoff: the zero-prob branch-pruning fast path is a no-op while any
action's floor is above 0, so sweeps revert to full-width tree walks.

| spot | info sets | resumed@iter | +iters | wall (solve step) | s/iter |
|---|---:|---:|---:|---:|---:|
| 11×11 d0 | 5,611,123 | 1380 | +200 → 1580 | 1146.3s | ~5.0s |
| 11×11 d1 | 5,611,123 | 1200 | +200 → 1400 | 1124.3s | ~5.0s |
| 11×10 d0 | 5,611,526 | 1160 | +200 → 1360 | 1002.1s | ~4.5s |
| 11×10 d1 | 5,611,526 | 1120 | +200 → 1320 | 1105.8s | ~5.0s |
| 10×10 d0 | 39,508,752 | 910 | +100 → 1010 | 8501.9s | ~85s |

The ~17× info-set ratio (39.5M vs 5.6M) roughly matches the ~17× s/iter
ratio (85s vs ~5s) — direct confirmation that trembling's cost is close to a
full unpruned sweep, not proportional to whatever fraction of the tree the
converged strategy still visits. 10×10 got half as many added iterations as
the 11-column spots for exactly this reason (cost, not diminishing need).
Total wall time: 11-column jobs ran in parallel (~20 min wall total for all
4); 10×10 ran alone (~2.36h solve + certify-export time). Combined GCP spend
was a small fraction of the ~10-25 USD guardrail — two short-lived on-demand
VMs, under 3 hours of combined wall time, no spot preemption risk taken
since the jobs were short enough not to need it.

### Acceptance results (before = currently-shipped pre-tremble charts)

All 5 re-exports verified bit-for-bit on structure: identical node/row
counts to the previously-shipped charts, `pts`/`own_reach` present on every
action/row (the latter newly exported — see `crates/truco-solver/src/bin/
solve.rs` `run_export_chart` — a genuine solver-computed σ̄-traversal own
reach, not the study lab's client-side path-product approximation), base
and deep files sharing an identical certificate as historically.

| spot | raw_eps before→after | purified_eps before→after | residue (info-set mass, before→after) |
|---|---|---|---|
| 11×11 d0 | 9.99e-6 → 5.246e-3 | 5.30e-6 → 3.15e-4 | n/a (not residue-limited) |
| 11×11 d1 | 9.82e-6 → 5.858e-3 | 4.53e-6 → 3.21e-4 | n/a |
| 11×10 d0 | 9.97e-6 → 4.695e-3 | 3.23e-6 → 1.89e-4 | n/a |
| 11×10 d1 | 9.86e-6 → 3.659e-3 | 3.30e-6 → 1.30e-4 | n/a |
| 10×10 d0 (`--allow-residue`) | 2.483e-4 → 2.380e-3 | 2.361e-4 → 1.223e-3 | 3.923% → 3.904% (essentially unchanged) |

Every purified eps stays under 1.3e-3 absolute (well under 0.13pp
match-equity exploitability) — "small growth from the tremble floor,"
exactly the acceptable range the task guardrail anticipated. 10×10's
residue (the pre-existing Q-gap mass that required `--allow-residue` even
before this task, inherited from its "provisional" grade) is essentially
unchanged, confirming trembling didn't introduce a new convergence problem
on top of the already-known one.

Study lab trainedness flags (self-loss > `assert_qgap_pp`=5.0pp, own-reach <
1e-3), reach-weighted over every row in base+deep combined:

| spot | rows | self-loss flagged before→after | own-reach flagged (purified `p`) before→after | own-reach flagged (RAW `p`) before→after |
|---|---:|---|---|---|
| 11×11 d0 | 190,067 | 12.97% → 0.58% | 51.24% → 25.71% | 51.18% → **0.00%** |
| 11×11 d1 | 190,067 | 8.25% → 1.36% | 44.79% → 25.40% | (not measured) |
| 11×10 d0 | 190,470 | 14.43% → 1.62% | 53.85% → 37.18% | 53.77% → 10.90% |
| 11×10 d1 | 190,470 | 15.87% → 1.69% | 55.93% → 40.77% | (not measured) |
| 10×10 d0 | 215,046 | 7.55% → 4.03% | 32.67% → 9.82% | (not measured) |

Self-loss (is the strategy AT an already-reachable node any good) collapses
8-16× at the 11-column spots and ~1.9× at 10×10 (half the iteration dose,
7× the tree). Own-reach measured on the shipped, purified `p` only
partially improves — expected, not a shortfall: purification correctly
re-zeros genuinely dominated actions after averaging, and a descendant of a
correctly-pruned branch has near-zero reach in the TRUE equilibrium
regardless of training quality. Own-reach measured on the RAW
(pre-purification) average — the quantity the tremble floor directly
targets — collapses to exactly 0% at 11×11 and drops 5× at 11×10 (deeper
histories mean the `(ε/|A|)^depth` floor itself legitimately falls under
the fixed 1e-3 threshold at depth 4, a math consequence of an absolute
threshold vs a depth-compounding floor, not a defect). Full discussion in
`RESEARCH_NARRATIVE.md` 2026-07-11.

The concrete garbage probe from `RESEARCH_NARRATIVE.md` open question 0
(11×10 d1, history `[33,0,0]`, mão holding 5♥5♠+4) is fixed: HIDE actions
(q=−1.0, certain loss) dropped from p≈0.21 each to raw_p≈0.86% each
(purified to p=0.0), PLAY actions rose from p≈0.29 to p=0.50 each.

All 10 files (5 spots × base + deep) copied into
`truco-frontend/public/study/`, verified valid JSON/gzip, node/row counts
matching. GCS artifacts (refined checkpoints, `.teach`, logs) under
`gs://truco-solver-runs/tremble-refine-20260711/`; both VMs deleted after
downloading (zero ongoing Compute Engine billing).

---

## Cost and Resource Observations

### Resource usage (c2-standard-8)
- CPU: 100% of one core (single-threaded)
- Memory: 12.8 GB (40% of 32 GB) — **over-provisioned by 2.5×**
- Optimal machine: c2-standard-4 (4 vCPUs, 16 GB) — cheaper, same speed

### Parallelism opportunity
The solver is currently single-threaded. 702 independent subgames (78 scores × 9 TCs) can be solved in parallel. On a 64-core machine with ~512 GB RAM:
- Run ~80 subgames simultaneously (512 GB / 6.4 GB each ≈ 80)
- 702 / 80 = ~9 batches
- Each batch: 6-8 hrs at ε = 0.01
- **Total: ~2-3 days**

---

## Optimizations To Benchmark Next

### High-impact

1. **Lower-score tree survey** — the `11×11` subgame is already well
   understood. The next real planning bottleneck is how much bigger the
   `10×10`, `8×8`, `5×5`, and `0×0` trees are once the raise ladder returns.

2. **Within-subgame parallelism** — the pipeline now parallelizes across
   independent jobs, but a single subgame solve still uses one core. If lower
   score states become much larger, the hot loop may need its own parallel
   story.

3. **Regret pruning** — still a plausible way to cut late-iteration cost if
   lower-score states spend a lot of time traversing obviously bad actions.

4. **Advanced tabular variants (DCFR+ / PDCFR+)** — promote the successful
   autoresearch DCFR+ idea into the main solver, then benchmark it and a
   predictive DCFR+ variant against the current hand-tuned DCFR baseline before
   considering any neural/model-free DeepPDCFR-style implementation.

### Medium-impact

5. **Warm starts across neighboring scores** — nearby score states may be
   similar enough to reduce iterations materially.

6. **Smaller info set key encoding** — current hashing may still have room for
   improvement if memory or cache behavior becomes a bottleneck.

### Lower priority

7. **Alternative weighting schedules** — linear CFR and other CFR-family
   variants are still worth trying if lower-score behavior diverges from the
   `11×11` results.

---

## Open Benchmarks Needed

- [x] **Exploitability every N iterations** — support landed via `expl_every`
- [x] **DCFR convergence at 11×11, TC 0** — measured through 120 iterations
- [ ] **Tree size survey (treesize mode)** — measure nodes/info-sets at 10×10, 8×8, 5×5, 0×0
- [ ] **DCFR convergence at 10×11 and 10×10** — confirm whether the same algorithm choice still wins
- [ ] **DCFR+ / PDCFR+ tabular comparison** — add first-class solver variants and benchmark against DCFR before attempting neural DeepPDCFR-style work
- [ ] **Full 11×11 all 9 turnup classes** — measure whether turnup classes behave similarly enough to simplify planning
- [ ] **Parallel speedup inside one subgame** — benchmark whether the hot loop benefits from additional rayon work
- [ ] **Lower-score convergence** — does exploitability converge at the same rate as 11×11?
- [ ] **Serialization size** — actual compressed/uncompressed strategy file sizes after a real solve

---

## 2026-07-16 — Final refined Study artifacts

All five completed tremble/detremble solutions were re-exported from their
final checkpoints with one exact raw/purified best-response pass and a
full-tree per-infoset BR-gap table. Epsilon is in the solver's `[-1,1]` utility
scale; multiply by 50 for match-win-equity percentage points.

| spot | raw eps | purified eps | raw residue | BR records | BR gzip |
|---|---:|---:|---:|---:|---:|
| 11x11 tc0 d0 | 0.0049562 | 0.0003814 | 2.137% | 3,144,536 | 13.80 MiB |
| 11x11 tc0 d1 | 0.0057960 | 0.0004578 | 2.206% | 3,057,578 | 13.29 MiB |
| 11x10 tc0 d0 | 0.0042378 | 0.0001724 | 2.951% | 2,849,517 | 12.53 MiB |
| 11x10 tc0 d1 | 0.0030554 | 0.0001092 | 2.664% | 2,697,912 | 12.73 MiB |
| 10x10 tc0 d0 | 0.0029086 | 0.0014782 | **3.936%** | 22,856,892 | 156.25 MiB |

The successful export pass ran 26m40s on an `n2-highmem-8`; including two
fast-failing infrastructure attempts, the temporary instance's total compute
charge was approximately $0.35 and it was deleted immediately afterward. The
published immutable release contains 35 artifacts plus its manifest, 326.66
MiB total. The 10x10 exact exploitability and BR table are valid outputs of the
selected profile, but its source teacher export exceeded the normal 3% residue
guard and remains provisional for interpretation.

## 2026-07-16 — Cross-turn-up warm-start canary (11×11 TC0→TC1 d0)

Question: can a tremble/detremble-refined policy from one blocker class cheaply
repair the off-equilibrium behavior of another class at the same score? The
opt-in transfer rewrites only `InfoSet.turnup_class`, matches actions by
identity, and preserves regret plus average-strategy accumulators. Rows that do
not exist in the donor remain fresh. The production-shaped canary used all
deals, the final TC0 refined checkpoint at iteration 8180, the retained TC1
treepack, and exactly one target iteration so both profiles received an exact
global BR pass.

| profile | raw eps | purified eps | weighted BR-gap mean | weight >1pp | weight >5pp | weight >10pp |
|---|---:|---:|---:|---:|---:|---:|
| native TC1 (1680 iters, no tremble repair) | 0.00000933 | 0.00000668 | 0.7290pp | 8.1220% | 4.4687% | 2.1147% |
| mapped refined TC0 + 1 TC1 iter | 0.00273153 | 0.00043611 | 0.00937pp | 0.00581% | 0.0000036% | 0.0000016% |

The mapped profile cuts the weighted local gap 77.8× and essentially removes
high-gap reachable mass. Its global raw/purified epsilon is worse than the
native profile, as expected when the blocker multiplicity changes, but the
purified value is only 0.0218 match-equity percentage points and is comparable
to the already-accepted refined TC0 Study profile. The one target solve took
36.7s; the complete successful VM lifecycle (install, release build, 3.3GB of
input downloads, two teacher exports, two exact certifications, upload) was
5m07s on Spot `n2-highmem-8`, about $0.02 compute. Three earlier fast-failing
diagnostic attempts kept total compute to only a few cents (comfortably under
$0.10); the final VM and auto-delete 200GB disk were deleted.

The failures exposed a separate artifact-compatibility hole: the retained TC1
strategy had 10.69M saved probability slots while the current retained treepack
had 5.37M legal-action slots. Export previously paired them positionally and
eventually reported only `teach truncated`. Export now projects by exact action
identity, renormalizes retained mass, uses uniform if the target contains an
action absent from the saved policy, validates every teacher tensor length,
flushes/syncs, and checks the exact
wire length before rename. Focused projection and round-trip tests cover this.

Evidence (summaries, full BR tables, logs, checksums):
`gs://truco-solver-runs/cross-turnup-canary-20260716/results/11x11-tc0-to-tc1-d0-attempt4/`.

## 2026-07-16 — Proof-scoped ex-ante action pruning

The attachment's broad hand-strength/equity proposal remains rejected: weak
raises can be equilibrium bluffs, and a final-card reveal can change the
opponent's raise range. The focused rules pass found two narrower, strategy-
independent certificates:

- A second mover's face-down play in rounds 2/3 is weakly dominated by showing
  the same card. The play resolves the round; if hiding loses round 2, the
  second mover could not have won round 1 and the hand ends. A round-3 leader's
  hide is removed only when the responder cannot raise and has only its forced
  final card. A round-2 leader can still use concealment to affect which of two
  cards the opponent plays, including during mao de onze, so blanket "never
  hide at 11" is deliberately not implemented.
- When the responder to a final-round raise lost round 1 and led either hidden
  or the globally weakest card, accepting guarantees a hand loss: the raiser
  must answer face up, and a tie also goes to the round-1 winner. `AcceptRaise`
  is dominated by folding at the old stake. A legal re-raise remains because
  it can win through fold equity; at 9x9 the pre-existing match-deciding-stake
  rule removes that re-raise too, leaving exactly `Fold`.

Tests pin all of those boundaries, including the unsafe final-leader signaling
case and the lower-score re-raise. The final 300-deal 7x7->8x8 warm A/B used
identical source regrets and match values:

| build/actions | nodes | info sets | 100-iter eps | value d0 | wall |
|---|---:|---:|---:|---:|---:|
| prior full actions | 2,321,925 | 829,386 | 0.000172 | 0.03797525 | 8.6s |
| proof-pruned | 1,252,901 | 628,742 | 0.000182 | 0.037973 | 6.1s |

The value delta is about `2.3e-6`, well inside both finite-run exact-BR
certificates. The all-deal census shows a much larger raw-history reduction
than information-set reduction because many removed histories collide into
info sets that remain reachable elsewhere:

| tier | prior nodes -> pruned nodes | node shrink | prior infos -> pruned infos | info shrink |
|---|---:|---:|---:|---:|
| mao de onze | 49,588,022 -> 21,909,190 | 2.26x | 5,611,123 -> 5,177,703 | 1.08x |
| {1,3} | 352,297,258 -> 179,876,874 | 1.96x | 39,508,752 -> 36,475,160 | 1.08x |
| {1,3,6} | 1,109,988,460 -> 602,162,740 | 1.84x | 129,144,643 -> 120,004,455 | 1.08x |
| {1,3,6,9} | 2,873,154,420 -> 1,599,822,324 | 1.80x | 341,656,035 -> 317,704,361 | 1.08x |
| {1,3,6,9,12} | 6,704,430,530 -> 3,802,814,334 | 1.76x | 812,865,845 -> 756,839,741 | 1.07x |

The paid decision run was the actual all-deal 10x10->10x9 TC0/d0 neighboring
warm workflow at `epsilon=0.01`, trembling off. It reached the same 40-iteration
stopping point and a slightly better certificate, but was slower and used more
peak memory than the retained full-action control:

| actions / loader | infos | final eps | internal solve | process wall | peak RSS |
|---|---:|---:|---:|---:|---:|
| prior full actions | 39,508,752 | 0.009825 | 1,500.6s | 28m02.9s | 58,372,788 KiB |
| proof-pruned | 36,475,160 | 0.009622 | 1,714.9s | 30m49.0s | 64,454,448 KiB |
| proof-pruned + direct-to-dense warm load | 36,475,160 | 0.009622 | 1,613.7s | 29m02.8s | 45,700,160 KiB |

Decision: keep the proof-backed rules, but assign **no solve-cost multiplier**.
The removed histories were already mostly skipped by CFR's exact-zero opponent
branch pruning, while the warm path's temporary target `StrategyTable` plus
fully deserialized source table controls VmHWM. Steady-state RSS fell near 16
GiB after transfer, but the 61.5-GiB high-water mark still required the old
75-GiB worker.

The first representation follow-up now passes. Checkpoint warm starts allocate
the table-indexed `DenseAccum` directly and project source rows into it, instead
of constructing a complete empty target `StrategyTable` first. Same-band,
cross-turn-up and historical mao-remap unit fixtures produce exactly the same
regret/average arrays as the compatibility path; the production epsilon
trajectory matched the old path at iterations 1/10/20/30/40 and the final game
value matched exactly. Against the identical proof-pruned control, wall improved
6.1% and peak RSS fell 29.1%, from 61.47 to 43.58 GiB. A 55-GiB worker gives
about 20% memory headroom and is 27.5% cheaper per modeled hour than the old
75-GiB worker, making this one shallow job about **1.46x cheaper** end to end.
It is also about 25% cheaper than the retained pre-dominance control after
right-sizing, despite being 3.6% slower in wall time.

This direct path still deserializes the full source `StrategyTable`; eliminating
that remaining 43.6-GiB transfer peak requires a row-streamable checkpoint
format. Do not apply the shallow 55-GiB result to the deeper tiers without a
deep-band memory scout. A deliberately uniform extrapolation would move $31K
only to about $23K, still far above the target, and is not credited in the
evidence-based bracket.

The current evidence-backed strict `epsilon=0.01` full-grid projection therefore
stays about **$31K spot** after generic neighboring warm starts, not lower. The
current materialized whole-grid exact-BR evaluator is separately about a $10K
operation: at 10x10, full build + both BR passes measured 461.8s versus 22,932s
for the 900-iteration tight solve, or 2.0% of the raw ~$505K lattice. The score
DAG itself is cheap; materializing every local hand tree is the expensive part.
Plan 79 now separates a <=$2 sampled reach/error allocator (prioritization, not
a certificate) from a compact DFS/depth-wise exact-BR prototype intended to
remove solver-arena RAM from final certification.

Production artifacts:
`gs://truco-solver-runs/cost-opt-20260716/dominance-warm-v3/10x10-to-10x9-tc0-d0/`,
`gs://truco-solver-runs/cost-opt-20260716/dominance-census-v3/tc0-d0/`, and
`gs://truco-solver-runs/cost-opt-20260716/direct-dense-v1/10x10-to-10x9-tc0-d0/`.
The scouts plus short setup failures remained below their separate $2 feature
caps (under $1 total compute); completed VMs and boot disks were deleted
after artifact verification.

## 2026-07-16 — Cross-score Study profile-transfer canaries

Four all-deal TC0 canaries transferred the final refined 11x10 checkpoint to
near/far scores and both dealers. Every candidate received an independent exact
global certificate and full per-infoset BR table. The donor average is excellent
off path, but score-dependent terminal utilities make its on-path suitability
non-monotonic:

| target | native raw/purified eps | transferred raw/purified eps | transferred weighted BR gap | selection |
|---|---:|---:|---:|---|
| 11x0 d0 | 0.00000892 / 0.00000892 | 0.000195 / 0.000195 | 0.0110pp | transfer |
| 11x0 d1 | 0.00000970 / 0.00000970 | 0.000266 / 0.000266 | 0.0122pp | transfer |
| 11x9 d1 | 0.00000961 / 0.00000265 | 0.006037 / 0.003194 | 0.0943pp | transfer |
| 11x9 d0 | 0.00000995 / 0.00000415 | 0.019508 / 0.016147 | 0.1392pp | native fallback |

The 11x9/d0 candidate barely changed across 22 target iterations in its
10-minute cap (about 0.019545 -> 0.019508), showing that a short CFR tail cannot
quickly erase the preserved donor average. Its local BR quality was nevertheless
about 88x better than native (0.139pp versus 12.29pp). The production expansion
therefore spends at most 90 seconds probing each transfer, selects it only at
raw eps <=0.01, purified eps <=0.004, and weighted mean BR gap <=0.25pp, and
otherwise exports the converged native profile with an honest BR table. The
four canaries plus one archive-path setup retry cost roughly $0.10 compute;
all VMs and boot disks were deleted after evidence upload.

## 2026-07-16/17 — Four follow-up cost gates: fold closure, allocator, streaming, compact BR

### Observable-card forced-fold generalization: zero structural gain

The proposed extension replaced the final-raise caller's globally weakest-card
test with the weakest opponent remainder possible across every hidden deal
consistent with its information set. The implementation used only own cards,
turn-up multiplicity and public plays—never the traversal's actual hidden
opponent hand—and an info-set-consistency test guarded that boundary.

Exact 300-deal counts did not change at either end of the raise ladder:

| state | before/after nodes | before/after info sets |
|---|---:|---:|
| 0x0 / TC0 / d0 | 7,905,387 | 3,859,486 |
| 9x9 / TC0 / d0 | 374,605 | 193,769 |

This is a proof-level closure, not merely a failed sample. Making `Plain(1)`
the weakest possible unseen opponent card requires exhausting all `Plain(0)`
copies from the caller's hand and public plays, but that card allocation cannot
also produce the required lose-round-1/win-round-2 history. Higher candidate
ranks require exhausting more lower cards than the five observable card slots
can contain. The existing `Plain(0)` rule therefore covers every reachable case
in this family; the experimental code was reverted. Cost: $0.

### Sampled whole-match allocator: useful ordering, deliberately not equity

`solve allocation-scout` now:

1. streams compact saved average policies (optionally adding the exact
   player-label symmetry needed after dealer alternation);
2. evaluates fixed strided deal panels at one representative of all seven tree
   bands, with explicit `all`/`first`/`all-except-raise` missing-action policy;
3. reuses the profile hand-outcome kernels through the complete 12x12 score
   DAG; and
4. computes both compact one-hand deviations and weights them by expected
   profile visits.

The sum is called **priority error mass**, not epsilon. A match can visit many
hands, so this cumulative quantity can exceed one; profile reach also differs
from adversarial reach. Converting it to equity pp would be wrong. Three
interleaved panels expose deterministic sampling sensitivity.

The first $0 run projected a 300-deal 8x8/TC0/d0 neighboring checkpoint across
the match. TC0 is only 10% of natural turn-up mass and is renormalized here;
this is a feature gate, not a nine-TC result. At 96 deals / three panels with
`all-except-raise`:

| band | mean visits | priority share | panel error-mass range |
|---|---:|---:|---:|
| full `{1,3,6,9,12}` | 7.480 | **32.88%** | 1.140-1.256 |
| `{1,3,6,9}` | 5.059 | **25.43%** | 0.823-1.005 |
| `{1,3,6}` | 2.773 | 14.99% | 0.484-0.609 |
| mao, P0 at 11 | 1.203 | 12.66% | 0.399-0.529 |
| mao, P1 at 11 | 0.604 | 6.56% | 0.182-0.335 |
| `{1,3}` | 0.499 | 5.04% | 0.146-0.222 |
| 11x11 | 0.239 | 2.46% | 0.064-0.122 |

Total priority mass ranged 3.248-3.990 across panels. A separate 24-deal
`all` fallback sensitivity put 52.1% in the full ladder and 22.6% in
`{1,3,6,9}`; `all-except-raise` at the same size put 32.7% / 26.9% there.
Thus the robust result is that newly legal deep-band raise behavior is the next
place to benchmark, while absolute allocation/equity remains uncertified.

### Row-streamed warm source: 24% cheaper shallow job, output now controls peak

The current positioned checkpoint format was already metadata + row count +
individually serialized rows. `CheckpointStream` now validates those rows and
projects them into table-indexed dense accumulators one at a time. Legacy
artifacts retain the full loader. Same-band, cross-turn-up and historical mao
remap fixtures produce exactly the same dense regret/average arrays as both the
old table path and direct-dense path.

The paid A/B repeated the identical all-deal 10x10 -> 10x9 TC0/d0 warm solve,
epsilon 0.01, no trembling:

| source loader | iter / eps | value d0 | internal solve | process wall | peak RSS |
|---|---:|---:|---:|---:|---:|
| full source table -> dense | 40 / 0.009622 | 0.198171 | 1,613.7s | 29m02.8s | 45,700,160 KiB (43.58 GiB) |
| streamed source rows -> dense | 40 / 0.009622 | 0.198171 | 1,696.0s | 30m49.4s | **33,133,128 KiB (31.60 GiB)** |

The solve phase itself stayed near 16 GiB. VmHWM occurred after convergence,
when returning/saving the solution rebuilt a full hash-table strategy while
dense accumulators were still alive. That makes 40 GiB, not 24 GiB, the honest
right-sized worker. At the catalog spot rates used by the census, 55 GiB costs
$0.2277/hr and 40 GiB $0.1629/hr; after charging the 6.1% wall regression, the
streamed job is still about **24% cheaper** ($0.0837 vs $0.1102 modeled compute
for this process). A uniform deep-tier extrapolation is forbidden; even if it
held, the prior ~$23K representation-only thought experiment would move merely
to ~$17-18K. The next lossless gate is direct dense-row checkpoint/strategy
serialization, removing the final dense+hash overlap.

Artifact: `gs://truco-solver-runs/cost-opt-20260717/stream-warm-v1/10x10-to-10x9-tc0-d0/`.

### Compact exact BR: local oracle equality

`solve compact-br` performs a space-for-time best response directly over
dynamic `TraversalState` DFS. It resolves responder information sets from the
deepest history upward, retaining only compact average-policy rows, one depth's
counterfactual action aggregates, and chosen actions. It never allocates the
solver arena or regret arrays.

Unit controls at 12 and 300 deals match profile value and both best responses.
On an actual 300-deal 8x8/TC0/d0 checkpoint:

| evaluator | profile P0 | BR0 | BR1 | epsilon | eval wall |
|---|---:|---:|---:|---:|---:|
| compact DFS | 0.035165212067 | 0.051653634065 | -0.032744338863 | 0.009454647601 | 2.791s |
| materialized arena | 0.035165212063 | 0.051653634067 | -0.032744338865 | 0.009454647601 | 1.509s |

The largest absolute value difference is `1.85e-10` (compact policy rows are
stored as normalized `f32`); the algorithmic oracle test using the same policy
representation is within `1e-10`. Compact is 1.85x slower locally, establishing
the intended CPU-for-RAM trade. The all-deal production result follows below.

### Compact exact BR all-deal gate: memory passes; its epsilon exposed a checkpoint-portability hazard

The <=$2 scout (`truco-compact-br-tc0`, Spot `n2-highmem-2`, 16 GiB) ran
`solve compact-br` over all 140,118 deals of 10x10/TC0/d0 against the actual
`10x10-full-20260709` checkpoint:

| metric | compact all-deal 10x10 |
|---|---:|
| eval wall (profile + BR0 + BR1) | 138.0s + 603.1s + 672.8s = 1,414.0s |
| process wall | 24m02.9s |
| peak RSS | 6,240,416 KiB (**5.95 GiB**) |
| chosen BR rows | 17,551,727 / 18,164,469 |
| DFS visits | 1,778,863,386 |
| threads | 1 (99% of one core) |

The memory/time gate **passes**: exact whole-game best response at the
largest shipped scale without the solver arena, inside a 16-GiB worker
(the materialized evaluator needs a >=40-GiB class for the same checkpoint,
and its loader panics outright on old artifacts — below). Per evaluation the
trade is roughly cost-neutral at the census's catalog-rate family (about
3.1x the materialized oracle's 461.8s, on a worker about 3.8x cheaper per
hour), so the modeled whole-grid exact-certification pass moves only ~$10K ->
~$8K. The real gain is structural: certification no longer sets the fleet's
memory floor, and the pass is single-threaded today so it can still be
parallelized. VM and boot disk were deleted after retrieval; total scout cost
was a few cents. Artifact:
`gs://truco-solver-runs/cost-opt-20260717/compact-br-v1/10x10-tc0-d0/`.

**The printed epsilon is NOT a certificate of the shipped checkpoint.** The
run reported `epsilon=0.016309` (gain0 0.015824, gain1 0.016793) with
`missing_profile_decisions=0`, while the same checkpoint certified
`epsilon=0.000248` at solve time — a 65x discrepancy. The cause is the tree,
not the evaluator: the scout binary (`894a753`) bakes in the 2026-07-16
proof-scoped prunes, so the 2026-07-09 policy was silently projected onto a
smaller tree, with saved mass on now-pruned actions renormalized across the
survivors. A $0 local discrimination pinned this down on 300-deal 8x8/TC0/d0:

| checkpoint | evaluated on | projection | epsilon |
|---|---|---|---:|
| current binary, 60 iters | its own (pruned) tree | — | 0.007912 (compact = arena to 5e-11) |
| pre-dominance binary, 80 iters, own-tree eps 0.008975 | pruned tree | renormalize | 0.216281 |
| same | pruned tree | remap (below) | 0.195955 |
| pre-dominance binary, 350 iters, own-tree eps 0.000485 | pruned tree | remap | 0.192743 |

The 350-iteration row falsifies the "early-iteration noise" hypothesis: the
old equilibrium genuinely keeps ~36% average mass on hidden plays at the
affected rows (82k mass over 224k projected row visits, essentially unchanged
by 4x more convergence). Concealment/tie mixing is load-bearing in
pre-dominance equilibria, and the old policy's continuation rows below the
newly-forced face-up branches were never trained for that reach. Weak
dominance guarantees a good continuation exists after substituting the
dominating action; it does not make a fixed old policy's continuation good.
**Consequence: no local row projection can port a pre-dominance checkpoint
onto the pruned tree.** The shipped 10x10 policy remains certified 0.000248
on its own tree; 0.016309 is the price of naively projecting it. Old
artifacts need either an old-tree evaluation mode (not built) or a
warm-started re-solve on the new tree, which is measured to heal the
projection (10x10 -> 10x9 reached 0.009622 in 40 iterations).

Two guardrails landed from this: `--project-dominated remap|renormalize`
(default `remap`: pruned `PlayFaceDown(c)` mass moves onto `PlayFaceUp(c)`,
certificate-pruned `AcceptRaise` mass onto `Fold`; measured never worse than
renormalize, exact no-op on matched trees, unit-tested at both boundaries)
and a `COMPACT_BR_PROJECTION` diagnostic reporting remapped/dropped mass —
the loud mismatch signal that `missing_profile_decisions=0` failed to be.
The arena `--control` path now detects mismatched action rows (459,790 on
the local case) and skips cleanly instead of panicking mid-pass with a slice
index error in `best_response_value`. The all-deal compact-vs-arena equality
leg remains open until a current-tree all-deal checkpoint exists to compare
on; the 300-deal matched-tree agreement (5e-11) plus the unchanged tiny/300
oracle tests are the equality evidence today.

## 2026-07-17 — How similar are neighboring equilibria? (`compare-policies`, $0)

`solve compare-policies` streams two saved average strategies and joins them
on the shared info-set key space (score is not part of the key, so same-band
cross-score joins are exact; cross-TC reuses the warm-start turnup-field
remap). Statistics are deliberately UNWEIGHTED over table rows — a
descriptive map of where policies differ, not a reach-weighted behavioral
distance and not an exploitability substitute. TV is total variation
distance per row; "pure" means max action probability > 0.99 on both sides.

Both mão-band pairs ran locally in ~4s each on the converged native artifacts
(5,611,526 rows per side):

| pair | matched | median TV | mean TV | rows TV<=1e-3 | argmax agree | pure-both | pure agree |
|---|---:|---:|---:|---:|---:|---:|---:|
| 11x10 vs 11x9 (tc0/d0, cross-score) | 100% | 0.000000 | 0.0168 | 93.7% | 96.0% | 18.7% | **99.99%** |
| 11x10 tc0 vs tc1 (d0, cross-TC remap) | 99.97% | 0.000000 | 0.0166 | 93.5% | 96.2% | 19.0% | **99.99%** |

Depth structure is the interesting part. Divergence concentrates in the few
shallow, high-reach rows and differs by pair type:

| depth | cross-score mean TV / agree | cross-TC mean TV / agree | rows |
|---:|---:|---:|---:|
| 0 | **0.140** / 87.6% | 0.003 / 99.8% | 403 |
| 1 | 0.098 / 89.8% | 0.046 / 94.8% | 403 |
| 2 | 0.087 / 88.5% | 0.030 / 95.7% | ~4.9k |
| 3 | 0.079 / 89.7% | 0.082 / 89.4% | ~13.2k |
| 4-5 | ~0.024 / ~96% | ~0.023 / ~96% | ~329k |
| 6 | 0.016 / 96.1% | 0.016 / 96.2% | ~5.26M |

Readings:

1. Neighboring equilibria are ~94% row-identical, and wherever both are
   confidently pure they agree 99.99% — strong support for transfer-based
   solving economics and very encouraging for human distillation (the
   "crisp" part of optimal play is stable across neighboring states).
2. Cross-score divergence lives at the ROOT (depth 0 TV 0.140: the mão
   accept/fold decision responds to the opponent's score), while cross-TC
   divergence does not (depth 0 TV 0.003: the accept/fold barely depends on
   turn-up class). This is consistent with the transfer canaries: the same
   ~4-6% of shallow on-path rows that differ are what made the 11x9/d0
   transfer certify at 0.0195 despite 96% row agreement — exploitability is
   carried by a small, shallow, score-sensitive core, and CFR average
   inertia is slow to repair exactly those rows.
3. Cross-TC key spaces overlap 99.97% (1,825 rows each side unmatched,
   turnup-multiplicity structural).

Follow-ups if wanted: reach-weighted variant (weight rows by profile reach
from a compact DFS pass), and the 10x10 tc0-vs-tc1 pair (5.4 GB artifacts,
needs a small VM). Cost of this entry: $0 (local; ~2.2 GB of downloads).

## 2026-07-17 — Follow-up batch: legacy-tree evaluation, reach-weighted similarity, dense-direct artifact writes

### Legacy-tree evaluation mode: the old checkpoint library is measurable again ($0)

`compact-br --legacy-tree` (a `TreeRules` toggle on `TraversalState`)
reconstructs the pre-2026-07-16 tree — score-aware raise pruning only, no
proof-scoped prunes. Gate: the pre-dominance 300-deal 8x8 checkpoints
reproduce their solve-era numbers exactly on the reconstructed tree, with
zero remapped/dropped projection mass (the fingerprint that the tree is
right):

| checkpoint | solve-era eps / value | compact --legacy-tree |
|---|---:|---:|
| 80 iters | 0.008975 / 0.008279 | 0.008974524343 / 0.008278925494 |
| 350 iters | 0.000485 / — | 0.000485450614 |

The production self-certification of the all-deal 10x10 checkpoint on its
own (legacy) tree closed the compact-vs-solve-era equality loop at full
scale: **epsilon 0.000248280260** against the fleet's printed 0.000248, and
profile value 0.055711349962 against the solve-era sidecar
0.05571134996183 — twelve digits of agreement between two independent exact
BR implementations (the solve-time arena oracle and compact DFS), with zero
remapped/dropped mass. Both unilateral gains are now recorded (gain0
0.000240, gain1 0.000256). The legacy tree costs more DFS than the pruned
one: 1,867.6s eval (2.76B visits) versus 1,414.0s on the current tree.
The clean 10x10-as-is transfer certificate at 10x9 (same legacy tree, zero
projection mass, all deals) answers the zero-iteration-transfer question for
the {1,3} band: **epsilon 0.040996** (gain0 0.049287, gain1 0.032705),
profile value 0.181800 versus the solved 10x9 equilibrium's 0.198171. An
adjacent-score neighbor policy played as-is is 4.1x outside the 0.01 target,
so the measured 40-iteration warm tail is genuinely load-bearing — the
on-path score-sensitive core seen in the reach-weighted comparison is
exactly what transfer cannot supply. Contrast the mão band, where far-score
transfers certified at 2e-4: zero-iteration transfer is a per-band/per-spot
question that only a certificate can answer, which is precisely what a
certify-first sweep would automate. The mirrored 9x10 certificate is
symmetric: **epsilon 0.042507** (gain0 0.037421, gain1 0.047593), so both
adjacent-score directions land at 0.041-0.043 as-is. In this band the
certify-first sweep's economics are therefore: the certificate (~31 min on
a 16-GiB worker) costs about as much as the 40-iteration warm tail it would
hope to skip (~29 min) — worth automating only where transfer has a real
chance of passing (mão-band-like geometries), not as a blanket lattice
pass.

The production-scale 10x10 tc0-vs-tc1 comparison (39.51M rows per side,
99.97% key overlap after the turnup remap, 7.9 GiB peak on the same worker)
reproduces the mão-band cross-TC verdict: unweighted mean TV 0.0126 with
98.0% argmax agreement; reach-weighted mean TV 0.0240 with 97.2% agreement,
flat ~0.03 across on-path depths 0-3 (no score-style root spike); and where
both policies are near-pure — 26% of rows — argmax agreement is exactly
1.0000 over 10.3M rows. Neighboring turn-up classes are close to
interchangeable everywhere play actually goes, at both the smallest and the
largest solved band.

All four `legacy-cert-v1` cases completed on one Spot n2-highmem-2 in
~1h40m compute (~$0.07 including disk); the VM self-deleted after upload.

### Reach-weighted policy similarity: on-path, neighbors genuinely differ ($0)

`compare-policies --reach-weighted` traverses policy A's supported tree
(legacy rules for pre-prune artifacts) and weights each row by its visit
probability under A's own play. The same 11x10-vs-11x9 (tc0/d0) pair that is
~94% row-identical unweighted looks very different where play actually goes:

| metric | unweighted rows | reach-weighted |
|---|---:|---:|
| mean TV | 0.0168 | **0.1291** |
| argmax agreement | 96.0% | **85.9%** |
| mass with TV<=1e-3 | 93.7% | 51.9% |

Weighted per-depth means: 0.196 at the root (reach 1.0, the mão accept/fold
responding to the opponent's score), 0.12-0.18 at depths 1-3, and only
0.03-0.06 from depth 4 on. The score-sensitive core is small in row count
but carries roughly half the play probability — a sharper statement of why
the 11x9/d0 transfer certified at 0.0195 while 96% of table rows agreed, and
a direct budget signal: cross-score transfer pays for the tail of the
policy, while the shallow on-path rows are what short warm tails must
actually fix.

The cross-TC pair (11x10 tc0 vs tc1, remapped) completes the contrast:

| pair | weighted mean TV | weighted argmax agree | weighted mass TV<=1e-3 |
|---|---:|---:|---:|
| cross-score 11x10 vs 11x9 | 0.1291 | 85.9% | 51.9% |
| cross-TC 11x10 tc0 vs tc1 | **0.0225** | **97.2%** | 72.9% |

On-path, turn-up class barely moves play (root weighted TV 0.005; small
bumps 0.065-0.068 at depths 1/3 where card-strength distributions shift),
while score moves it a lot. This measures directly why productionized
cross-TC transfer is nearly free while cross-score transfer needs a repair
tail plus a certificate, and it ranks donors for any future certify-first
sweep: same-score/other-TC first, then adjacent-score.

### Dense-direct artifact writes (plan 79 phase 9c): lossless, byte-identical ($0 so far)

Two output-path memory sinks removed losslessly:

1. `save_checkpoint_iter` used to build an OWNED copy of every row and then
   `bincode::serialize` the entire multi-GiB file into one in-memory byte
   buffer before writing — at 10x10 scale, several GiB of avoidable overlap
   on every periodic checkpoint, not just at the end. It now sorts a
   borrowed row index and streams through a 4-MiB `BufWriter`.
2. `solve-tc` now writes the average-strategy artifact directly from the
   dense accumulators (`SolveConfig::strategy_output`) and skips the
   end-of-solve `StrategyTable` rebuild entirely
   (`SolveConfig::skip_return_table`) — dense and hash table never coexist.

Gates: a unit test pins both writers byte-identical to the historical
whole-struct serialization (including the zero-mass uniform-average row
case), and a deterministic 300-deal 8x8 re-solve reproduces the pre-change
strategy/checkpoint/gv artifacts bit-for-bit.

The <=$2 production A/B re-ran the identical all-deal 10x10 -> 10x9 TC0/d0
warm solve on a 24-GiB Spot worker (`dense-out-v1`, ~$0.06):

| output path | iter/eps trajectory | peak RSS | process wall |
|---|---|---:|---:|
| row-streamed source + table rebuild (9b) | 0.176369/0.029722/0.018258/0.012718/0.009622 | 31.60 GiB | 30m49.4s |
| dense-direct artifacts (9c) | **identical at every checkpoint** | **16.19 GiB** | 17m56.8s |

The solve-phase plateau is now the true peak: memory-class chain for this
job is 75 -> 55 -> 40 -> **24 GiB** across phases 9a/9b/9c. At the census
rate family (~$0.094/hr at 24 GiB) the completed job models at ~$0.028
versus ~$0.084 under the 40-GiB streamed loader — about 3x cheaper, though
roughly 1.7x of that is per-iteration host variance (169s vs ~285s per
10-iteration block on nominally identical 2-vCPU workers; the fleet
previously measured 1.3-1.5x zone variance), so the conservatively credited
factor is the ~1.7x memory-class ratio at equal wall. As with 9a/9b, no
deep-tier fleet credit until a deep-band memory scout: the honest uniform
thought experiment moves the ~$17-18K streamed extrapolation toward ~$10K,
and the evidence-based bracket stays $31K until deep tiers are measured.

### Deep-band exact BR scaling (0x0 full ladder, all deals): memory answered, values voided by a wrong table

The `deep-br-0x0-v1` scout ran `compact-br` over the FULL 0x0 ladder tree,
all 140,118 deals, with the 10x10 policy projected under `all-except-raise`.
The structural/scaling half — the reason the scout existed — is measured:

| metric | 10x10 ({1,3} band) | 0x0 (full ladder) | ratio |
|---|---:|---:|---:|
| eval wall (single thread) | 1,414.0s | 9,585.9s (2h40m) | 6.8x |
| DFS visits | 1.78B | 14.64B | 8.2x |
| chosen BR rows | 35.7M | 73.6M | 2.1x |
| max BR depth | 7 | 13 | — |
| **peak RSS** | 5.95 GiB | **8.13 GiB** | **1.4x** |

**Whole-game exact best response of the deepest band fits in 8.13 GiB.**
Certification memory scales with info-set count (2.1x rows -> 1.4x RSS),
not node count (8.2x visits), exactly as the compact design intended — a
16-GiB worker class covers the entire lattice's certification passes, and
the ~$8K modeled whole-grid certification cost now rests on measured deep
wall/memory rather than shallow extrapolation (single-threaded; deal-level
parallelism still available).

The VALUE half of the run is void and is recorded as a negative result: the
scout reused the 10x10 fleet's `match_values.bin`, whose low-score cells are
unsolved and default to probability 0.5 — which maps to a continuation
payoff of exactly 0 on the q scale, silently zeroing every profile/BR value
(`epsilon=0.000000`, 55.3M missing-policy decisions correctly reported, but
no warning about the table). Chasing the "complete" table surfaced the real
lesson: **no complete DP table exists anywhere** — producing it is the
whole project — so out-of-order deep-band VALUE certification is not merely
mispriced, it is undefined. In the actual bottom-up production DP this
never arises: by the time any state is solved or certified, every successor
cell is real by construction. `compact-br` now refuses to run when any
non-terminal successor cell is unsolved, turning ordering mistakes into
fail-fast errors. Scout cost ~$0.15; the memory/time answer above is
unaffected because traversal structure is value-independent.

Footnote uncovered by the same probe: the fleet table's `(10,10)` cell is
itself unsolved (the fleet's measured 10x10 values live in `.gv` sidecars
and were never written back), so the entire 10x9 scout family — the phase
2/9a/9b/9c warm A/Bs and today's 10x9/9x10 transfer certificates — was
consistently measured in a variant game with `mv(10,10)=0.5` instead of the
measured 0.528. Every relative conclusion (warm-start factors, memory
peaks, byte-identity, the 0.041-vs-0.01 transfer verdict) is unaffected
because both sides of every comparison shared the table; absolute 10x9
values carry the caveat. The guardrail now makes this class of gap loud.

## 2026-07-17 — Full 225-profile Study release fleet (profile transfer at scale)

The production expansion of the cross-score canaries: four Spot workers
(`c2-standard-8`, `n2-standard-8`, `n2-highmem-4`, `n2-highmem-8`,
us-east1) built all 225 (score, TC, dealer) Study spots — 11x0–11x11 x 2
dealers x TC0–8 plus 10x10 x TC0–8 — with per-spot 90-second transfer
probes, exact self-certification, and native fallback. Aggregated from
the 225 `COMPLETE.json` markers:

| source | spots | raw eps (median / max) | purified eps (median / max) | weighted BR gap pp (median / max) |
|---|---:|---:|---:|---:|
| transferred | 192 (85%) | 0.00124 / 0.00627 | 0.00046 / 0.00395 | 0.014 / 0.098 |
| refined (direct export) | 5 | 0.00306 / 0.00424 | 0.00037 / 0.00148 | 0.029 / 0.166 |
| native-fallback | 28 (12%) | 0.00001 / 0.01685 | 0.00001 / 0.01683 | 2.551 / 12.591 |

All 192 transfers cleared the acceptance gates (raw <=0.01, purified
<=0.004, weighted mean BR gap <=0.25pp). Fallback metrics describe the
shipped native profile (the worker re-exports from the native strategy
after rejecting a transfer); the rejected candidate's BR summary is
preserved per spot as `transfer-brgap-summary.json` in the audit prefix.

Operations: workers finished in 3h01m–4h34m wall against a 6.5h budget
sizing, so actual Spot spend landed well under the ~$5 target. One
preemption (`pt-fleet-11x4-7-0716`, 28 min in) was absorbed by a
marker-idempotent replacement VM that skipped the 12 finished rows. All
four VMs and boot disks were deleted automatically by the 8h
`maxRunDuration` + `instanceTerminationAction=DELETE` backstop after
guest poweroff — no manual cleanup was needed. Release
`20260717-full-225-v1` (225 spots, 675 sha256-manifested artifacts)
published atomically to `gs://truco-study-artifacts/releases/` by a
one-shot assembler VM; audit trail under
`gs://truco-solver-runs/profile-transfer-fleet-20260716/`.

### Purity of the shipped solutions (`policy-stats`, $0)

How crisp is optimal play? "Pure" = the row's maximum action probability
exceeds 0.99; play-weighting uses the policy's own on-path reach (legacy
tree, d0):

| solution | pure, all rows | pure, by play | mean max-prob (rows / play) |
|---|---:|---:|---:|
| 11x10 tc0 (5.6M rows) | 20.4% | **56.1%** | 0.609 / 0.873 |
| 10x10 tc0 (39.5M rows) | 26.8% | **52.4%** | 0.655 / 0.836 |

Mixing dominates the table but not the played game: half-plus of actual
play sits at (near-)pure decisions, and the cross-pair comparisons above
showed those pure rows agree across neighboring solutions at 99.99-100%.
The complement matters too: ~45% of play mass is genuinely mixed — the
part of optimal truco that resists crisp human rules (QUESTIONS.md Q4/Q5)
— and by the 0.999 threshold on-path purity drops to 50.7% (11x10) /
39.3% (10x10), so "nearly pure" softens with the cutoff. These figures now
back the guide's Abstractions-chapter similarity/purity section.

### Deep-band census on the current tree (0x0/TC0/d0, all deals, ~$0.07)

`count-tree` raw census on a 16-GiB Spot worker (52m30s count wall, 13.5 GiB
census RSS):

| band | nodes | info sets | vs 10x10 band |
|---|---:|---:|---:|
| 10x10 ({1,3}), current tree | ~200M (post-prune) | 39.51M | 1x |
| 0x0 (full ladder), current tree | **3,802,814,334** | **756,839,741** | ~19.2x info sets |

Against the pre-prune census (6.70B nodes / 812.9M info sets) the
proof-scoped prunes hold their shallow-band ratios at depth: 1.76x nodes,
~7% info sets. Solve-side implications, modeled from these counts before
the probe: dense accumulators ~35-40 GB, arena plus info-set metadata
80-120 GB — a 130-160 GB worker class, roughly $0.6-0.7/hr Spot, with
per-iteration wall projected ~10-19x the shallow band's. A bounded throughput/memory probe (`deep-solve-0x0-v1`, 160 GB, --jobs 4)
**failed by under-provisioning** and is recorded as a negative result. RSS
grew linearly from 4.3 GiB to **148.5 GiB in 43 minutes of tree build** (~2.7
GiB/min, still climbing) and livelocked against the 160-GiB (no-swap)
ceiling before a single CFR iteration printed. Two process errors caused it:
the worker was sized at the top edge of the *unverified* 130-160 GB model
this very probe existed to test, with no headroom; and with no swap and no
clean OOM abort, the box thrashed silently (the RSS sampler starved too)
instead of failing fast. The VM was killed manually; its 4h max-run-duration
would have been the backstop. Honest takeaway (a lower bound, since the build
never finished): **the 0x0 deep-band solve needs >150 GiB just to build and
start**, already above the 80-120 GB arena estimate, confirming the deep
tiers require a large-memory worker — the expensive part the $31K bracket
already anticipates. A re-run needs a 256-384 GB class worker AND a
fail-fast memory guard (or swap) so a mis-size errors in minutes instead of
livelocking for an hour; whether that spend is worth it is a judgment call
recorded in RESEARCH_NARRATIVE, not an automatic next step.

## 2026-07-17 — Asymmetric (per-player) raise pruning: the first structural deep-band win

The deployed raise-prune gates on `min(score)`: a raise is removed only when
the stake already decides the match for BOTH players. But the dominance
argument is per-acting-player. When the *acting* player p has `score[p] +
on_table >= 12`, p already reaches the target by securing `on_table` (win the
hand, or the opponent folds the raise and concedes `on_table` — which hands p
the match, so the opponent never folds); escalating is pure downside. So the
gate can key on the acting player's own score, not min(score). This prunes the
higher-scored player's dominated raises in lopsided states. Implemented as
`TreeRules::AsymmetricRaisePrune` (opt-in; strict superset of the deployed
prune since `score[acting] >= min(score)`).

**Structural A/B (`count-tree`, 300 deals TC0 d0).** Symmetric cells are
exactly unchanged (correctness sanity); lopsided cells collapse:

| score | band (min) | current infos | asym infos | ratio |
|---|---|---:|---:|---:|
| 9×9 | {1,3} | 193,769 | 193,769 | 1.000 |
| 6×6 / 3×3 / 0×0 | symmetric | — | — | 1.000 |
| 9×6 | {1,3,6} | 628,742 | 413,829 | 0.658 |
| 9×3 | {1,3,6,9} | 1,642,528 | 444,448 | 0.271 |
| **9×0 / 10×0** | full | 3,859,486 | 444,448 | **0.115** |

9×0, 9×3, 10×0 collapse to the *same* size: escalation is alternating, so the
higher player's cap bounds how deep the ladder is actually reachable.

**Value-preservation A/B (`solve-tc`, identical synthetic gradient match
values, ε=1e-4).** The dominance holds for any continuation values (it is
about reaching 12 = winning, encoded in the terminal payoffs):

| score | current v_d0 | asym v_d0 | Δ | current | asym |
|---|---:|---:|---:|---|---|
| 9×6 (4000 deals) | 0.21298177 | 0.21298133 | 4.4e-7 | 176s / 6.92M inf | 106s / 4.56M inf |
| 9×0 (1500 deals) | 0.62118004 | 0.62117395 | 6.1e-6 | 278s / 18.26M / 768MB | **22s / 2.10M / 85MB** |
| 11×6 mão (3000) | 0.62995441 | 0.62995441 | 0 (identical tree) | — | — |

Every Δ is far below the ε≈9e-5 both sides converged to (pure noise). Mão de
onze is untouched (the rule finds nothing extra to prune there). The 9×0 job
ran **12.6× faster on 9× less RAM**.

**Grid-level cost impact (full 121-cell TC0 sweep).** Per-tier tree totals,
under two cost models — linear (cost ∝ nodes, i.e. wall at a fixed RAM class)
and RAM×wall (cost ∝ nodes × info sets, i.e. a right-sized fleet):

| tier | node(wall) ratio | RAM×wall ratio | tight-ε $ | → RAM×wall $ |
|---|---:|---:|---:|---:|
| {1,3} | 1.000 | 1.000 | $54 | $54 |
| {1,3,6} | 0.813 | 0.690 | $4,127 | $2,846 |
| {1,3,6,9} | 0.642 | 0.487 | $53,956 | $26,269 |
| **full ladder** | 0.511 | **0.361** | $447,291 | $161,369 |
| **TOTAL** | **0.528** | **0.377** | $505,428 | $190,539 |

So the whole-grid solve cost drops to roughly **0.38–0.53×** (≈ half, plausibly
to a third with right-sized RAM). Applied to the current evidence-based
**$31K** bracket (ε=0.01 + warm starts): **~$12–16K**. The full-ladder tier —
89% of cost — takes the biggest cut (0.36–0.51×), which is exactly the point:
this is the first optimization that dents the deep bands rather than the
already-cheap shallow ones.

**Honest caveats.** (1) Iterations assumed tier-constant; the 9×0 job actually
converged in fewer (600 vs 800), so the estimate is if anything conservative.
(2) The ~9 symmetric-deep cells (0×0, 1×1, 2×2 and near-neighbors) do NOT
shrink and now dominate the residual full-ladder cost — this does not remove
the single-giant-VM requirement, it removes it from ~48 of 57 full-ladder
cells. (3) Ratios measured at TC0 only (tree structure is TC-independent, so
this should hold). (4) Weakly-dominated removal preserves game value (measured)
and yields a strategy that is safe to deploy (not raising is optimal where the
raise was dominated), though the equilibrium *set* can differ. This clears the
Phase-7 local gate; a production tier scout is the next step before fleet credit.

---

## 2026-07-17 — Lossless-prune audit: isomorphism + saturation candidates both close empty ($0, local)

Two candidate lossless reductions from the EXACT_SOLVING.md §6 thread-3 list,
audited via inspection + `count-tree` (300-deal subsets, tc0/d0, release
binary; baselines reproduced the published asymmetric-prune counts exactly).

**Residual card/suit isomorphism: ~1.0×.** Resolved by inspection, no run
needed: `InfoSet` (`info_set.rs:93`) is already the canonical quotient.
`AbstractCard::Plain` is suit-independent, manilhas carry a strict total
order, the starting hand is stored sorted, and every remaining key field
(`player`, `is_dealer`, `turnup_class`, `history`) is strategically necessary
(`is_dealer` collapse was the 11×10 exploitability wall). No residual
symmetry to dedup.

**Per-node match-saturation / forced-winner collapse: ~0 where it matters.**
The stake-saturation form is the deployed `min+stake>=12` prune
(`game_tree.rs:146`), i.e. degenerate. The forced-winner-by-card-strength
form is only lossless at frozen stake — live raises keep bluff/betting value
strategy-dependent — and the card-play envelope is <1% of a full-ladder cell:

| cell | ladder | nodes | info sets |
|---|---|---:|---:|
| 11×11 (mão, no raises) | frozen | 45,723 | 25,651 |
| 0×0 (full ladder) | live | 7,905,387 | 3,859,486 |

(300-deal subset; full-deal docs ratio 49.6M/6.70B ≈ 0.74%.) Even a perfect
collapse is capped below 1% of the expensive cells. Same runs re-confirmed
the symmetric wall: `--asymmetric-raise-prune` leaves 0×0/1×1 counts
bit-identical (7,905,387 / 3,859,486), while 9×0 and 9×3 both collapse to
880,273 / 444,448.

**Meta-result.** >99% of the expensive symmetric deep cells is live
betting/bluffing tree, which is strategy-dependent and therefore immune to
the entire lossless-pruning family (dominance, isomorphism, saturation).
Remaining exact-solve savings live in decomposition (CFR-D / safe subgame
re-solving) or representation (mmap/f32/distribution), not further pruning.
Recorded in `EXACT_SOLVING.md` §6 threads 3 and 6.

---

## 2026-07-17 — `accum-f32` reduced-precision accumulators: local A/B gate passes ($0, local)

Plan 84 Phase 1 / plan 79 Phase 6. Implementation: `DenseAccum`'s persistent
`regret`/`strategy` narrow to f32 under the `accum-f32` cargo feature;
transient per-sweep buffers (`pending`, `last`, parallel `LocalAccum`) stay
f64, so each cumulative slot sees exactly one narrowing per iteration
(widen-add-narrow at every fold site — a no-op in the default build).
Checkpoints/strategies stay f64-formatted via a generic `AccumElem` boundary
in `storage.rs`; default-build bytes unchanged. 97/97 unit tests in both
modes (the serial-vs-parallel equivalence test gains a feature-conditional
tolerance: the two paths narrow strategy sums at different points, so f32
trajectories are rounding-close, not identical).

A/B (release builds, 300 strided deals, tc0/d0, 200 fixed SyncCFR+ iters,
shared synthetic mv for 8×8):

| run | final exact ε | game value | peak RSS | wall |
|---|---:|---:|---:|---:|
| 11×11 f64 | 0.000086 | 0.324381 | 26.3 MB | 0.2s |
| 11×11 f32 | 0.000086 | 0.324381 | 26.9 MB | 0.2s |
| 8×8 f64 | 0.000286 | 0.000037 | 308.4 MB | 3.9s |
| 8×8 f32 | 0.000286 | 0.000038 | 294.6 MB | 4.4s |

ε identical at printed precision; game-value drift 1e-6 (contract ≤1e-5 abs
on cheap deterministic cases). The 8×8 RSS delta (−13.8 MB) matches the
predicted accumulator halving (~2M slots × 4B ≈ 16 MB expected); the rest of
RSS is tree/deal overhead, untouched by this feature — the headline RAM
fraction (~15–17% of a production job, more in decomposed subgame jobs)
must be validated at scale, not from these subsets. 8×8 wall +13%
(3.9→4.4s) — possible cast overhead, within local noise; re-measure at
scale before drawing conclusions.

Cross-build checkpoint compatibility: PASSES. Each binary warm-started the
other's 8×8 checkpoint via the streamed disk-backed path (f32←f64 and
f64←f32), both solving on cleanly with trajectories tracking at the
f32-rounding scale (iter-20 ε 0.020197 vs 0.020660 — the same-band
warm-start's average-reset semantics, identical in both builds).

**Still pending for adoption** (the plan-79 gate): the ≤1% relative ε drift
at a production-scale target — a paid scout, folded into plan 84 Phase 5.

---

## 2026-07-18 — CFR-D safe subgame re-solving: core lands, repair is near-perfect ($0, local)

Plan 84 Phase 3. Implementation described in `SOLVER_PLAN.md` (2026-07-18
entry): `subgame.rs` round-2 boundary decomposition via build-recursion
replay, `cfr.rs::best_response_boundary_values` (CBV oracle through the
certified 3-pass memo) + `resolve_subgame` (Burch–Johanson–Bowling
Terminate/Follow gadget over existing packed subtrees), `resolve.rs`
orchestration/composition/certification. 103/103 crate tests; `accum-f32`
mode compiles.

Integration results (8×8, 24 strided deals, synthetic complete mv, 60-iter
SyncCFR+ blueprint, tc0/d0):

| experiment | blueprint ε | corrupted ε | result ε |
|---|---:|---:|---:|
| repair (one subgame → uniform, re-solve from CLEAN boundary summary, 200 iters) | 0.007446 | 0.113004 (15×) | **0.007449** |
| resolve_all (all 576 subgames re-solved at 120 iters, composed) | 0.007446 | — | **0.005745** |

- **Repair recovers 99.997%** of the exploitability damage (to within 3e-6
  of the blueprint) using ONLY boundary CBVs + root reaches — subgame play
  is fully reconstructible from the boundary summary, which is the core
  CFR-D decomposition claim.
- **Composition is safe with room**: the composed profile beat the blueprint
  (its per-subgame re-solves ran longer than the blueprint solve), well
  inside the ε-budget bound `composed ≤ blueprint + resolve slack`.
- **Granularity**: 576 subgames from 24 deals; largest = 4 members / 1,863
  nodes / 921 info sets. The production promise is exactly this shape: peak
  memory per re-solve job scales with the largest SUBGAME, not the cell.

Caveats: prototype allocates full-size accumulators (structural stats, not
prototype RSS, are the Phase-5 cost inputs); blueprint-as-CFV-source
("oracle mode") — the trunk-CFR loop that produces CFVs without a full
solve is Phase 4; fixed-iteration re-solves. Next: CLI surface + the
production-scale 10×10 ground-truth run on GCP.

---

## 2026-07-23 — Phase 5 probe complete: 0×0 solves + certifies on one 128 GB commodity box ($7.55 of a $10 cap, 9 attempts)

Plan 84 Phase 5's question — does the decomposed deep path make the
symmetric wall cells commodity-box problems — is answered YES, with every
cost constant measured. Cell: 0×0 tc0/d0 (≡1×1 bit-for-bit), all 140,118
deals, 3,135 subgames, `--deep --jobs 16`, `accum-f32`, SYNTHETIC-complete
match-value table (`examples/gen_mv.rs` — 0×0's successors span the
never-solved grid; identical tree/arithmetic, so cost transfers exactly;
strategy content does not — this is a $/cell benchmark, not a shippable
0×0 policy).

**Measured (definitive numbers from attempts 5 and 10):**

| quantity | value |
|---|---:|
| init (trunk arena + 3,135 subgame builds + 15.7 GiB ckpt resume) | 88.4 min |
| per alternation round, (T,R)=(1,1), rebuild mode, PRE-seeds-fix | 70–85 min (near-OOM page pressure; see band below) |
| streaming certificate, SUBGAME-PARALLEL (bcb24234) | **10.6 min** (≥14× vs the sequential version that blew a 4h backstop) |
| raw ε after 3 effective rounds | **0.24585** (1/t curve: 0.9/3 ≈ 0.3 ✓) |
| peak RSS | 124.5 GiB on 128 GiB (fits; jobs=16 cert leaves ~zero headroom — use cert jobs≈8) |
| checkpoint | 15.74 GiB, atomic, resumed across 4 zones + 2 machine families + spot/on-demand |
| box | n2d-highmem-16 SPOT $0.24/hr (n2d pool was stocked when n2 was exhausted region-wide) |

**The three bugs the probe existed to find** (each caught for <$2):
1. Boundary-state capture OOM — a full `TraversalState` per crossing ≈
   78 GB at 0×0 (124.5 GiB OOM at 88 min). Fixed: ~40-byte replay seeds
   (`replay_crossing_state`, commit ce4f05d); deep gates bit-identical.
2. Sequential certificate — 15 of 16 cores idle, >2.4 h, blew the 4 h
   backstop on-demand. Fixed: subgame-parallel (bcb24234), 10.6 min.
3. Post-certificate artifact OOM (open, logged): `deep_solve` returns the
   composed profile as a ~757 M-row HashMap + clones rows; killed (signal
   9) AFTER the certificate printed. Fix pattern known: stream artifact
   rows like the phase-9c dense-direct writes. Blocks nothing measured.

**Per-round cost band and what it means.** The 70–85 min round figure was
measured before the seeds fix, at 124 GiB RSS (page-cache starvation), on
the slower n2 SKU. The parallel certificate — comparable node work — now
takes 10.6 min post-fix on n2d. True post-fix round cost is therefore
somewhere in **15–75 min**; one $0.50 spot run (3 rounds, resume) pins it.
Extrapolation to ε=0.01 (~150 effective rounds) at $0.24/hr:

| scenario | $/cell·tc·dealer |
|---|---:|
| current code, pessimistic (75 min/round) | ~$45 |
| current code, optimistic (15 min/round, post-fix) | ~$9 |
| + arena NVMe cache / keep-arenas + R>1 amortization (unbuilt) | ~$2–5 |

Even the pessimistic bound beats the never-provisionable 1.54 TB
monolithic box; the optimistic + engineering band is what makes the
whole-grid ~$3–4K Fermi (and the §teacher-grade ~10× ladder on top of it)
real. ε=2.5e-4 ≡ ~0.0125 pp equity costs ~10× the ε=0.01 iterations
(SyncCFR+ 1/t, measured at 10×10: 90 iters→0.0095, ~900→2.5e-4) and can
be bought LATER from the retained checkpoint at zero waste — iterations
compose additively (checkpoint-chunking gate is bit-exact).

**Ops runbook proven** (the fleet playbook the full grid needs): 6 spot
preemptions + 2 stockouts + 1 phantom create ridden out; incremental
uploader (log + stderr + checkpoint every 300s) made every death cheap;
DEEP_PHASE timers on stderr because Rust block-buffers stdout to files;
3-consecutive-miss VM-gone detection (list API false-negatives); machine-
family failover n2→n2d (separate capacity pools, n2d cheaper); on-demand
escalation only for the unfinishable-under-preemption stage. Attempt
ledger: $0.04 mv-panic, $0.66 + $0.71 OOM pair, $0.08 mercy-kill, $1.64
preempted-at-3.4h (completed all 3 rounds — checkpoint saved it), ~$0.10
phantom/stockouts, $3.60 on-demand 4h-cap (sequential-cert lesson),
$0.40 the run that landed everything. **Total $7.55 of the $10 cap.**

---

## 2026-07-21 — Phase 4: from-scratch trunk-CFR at 10×10 certifies 1.1e-3, gv to 5.5e-5 of ground truth (4 runs, $3.35)

Plan 84 Phase 4 complete: `solve trunk-solve` solves 10×10 tc0/d0 (all
140,118 deals, current tree, 36,475,160 info sets) with NO blueprint —
trunk + 495 subgames in alternation, tail-accumulated CBVs, gadget
recovery, exact-BR certificate. Same SPOT SKU/backstop pattern as Phase 3.

**The negative result that mattered (runs 1–3):** with intra-round
couplings (cached boundary values, subgame roots, CBV folds) reading the
AVERAGE profile, composed eps plateaued at ~0.018 across r90/r150/r300 —
iteration count was not the constraint. Run 3's three-certificate
diagnostic split the blame: raw_eps 0.0465 (the loop was the floor),
tail-CBV recovery 0.0176 (recovery was REPAIRING the weak loop), BR-CBV
recovery 0.0334 (BR against a weak blueprint is over-generous — confirming
the tail-CBV design choice). Cause: SyncCFR+ backups are defined against
the frozen CURRENT strategy; the average lags by ~half the averaging
window, so the coupled alternation chased a stale target.

**Fix** (`DenseAccum::current_row`, all three couplings switched): toy
probe (8×8/300 deals) went from 0.002644/(30,3,3) and 0.001147/(90,1,1)
to **0.000349 and 0.000239 vs monolithic-300 control 0.000354** — the
decomposed from-scratch solve now matches or beats monolithic at equal
effective iterations.

**Definitive run 4 (r150, T=1, R=1, final=200; 3h05m wall, 28.1 GiB RSS):**

| certificate | value |
|---|---:|
| raw eps (loop output, no recovery) | **0.001122** |
| tail-CBV recovery | 0.001266 |
| BR-CBV recovery | 0.001393 |
| game value d0 | **0.055766** vs teacher ground truth 0.055711 (Δ 5.5e-5) |

Acceptance (composed ≤ 0.01, gv within ~1e-3): met with 9× / 18× margins.
At this quality the raw profile IS the best certificate — recovery at
fixed 200 iters slightly degrades it (consistent with Phase 3's
blueprint 2.5e-4 → composed 1.7e-3). Multiplier: structural 3.65× by
design; measured wall vs a monolithic-150 extrapolation ≈ 2.4× including
the diagnostic's extra certificates/recoveries, ≈ 1.8× net.

**Cost ledger (4 runs):** r90 $0.48 + r300 $0.98 + r150-diag $0.78 +
r150-fix $0.97 + disk ≈ **$3.35**. The two "wasted" plateau runs bought
the lag bug — found only at production scale (the toy showed a 3× hint,
not a wall).

**Phase 5 inputs now measured:** largest re-solve unit 1.0M nodes / 197k
info sets (0.29% of cell); from-scratch decomposed quality ≈ monolithic;
RSS 28 GiB dominated by full trees + full-size prototype accumulators
(both go away with trunk-only arenas + per-subgame remapping).

---

## 2026-07-18 — CFR-D at production scale: 10×10 ground-truth run, repair to 1.6e-7 of blueprint ($3.45 total)

Plan 84 Phase 3 tail complete. `solve resolve-subgames` ran the full CFR-D
pipeline at 10×10 tc0/d0, ALL 140,118 deals, against the real fleet
blueprint (`10x10-full-20260709/d0` strategy artifact, mv from the same
run) on one `n2-custom-2-76800-ext` SPOT worker (us-east1-b, $0.314/hr,
`--max-run-duration` + `instance-termination-action=DELETE` backstops).

**Definitive run (attempt 4, `--legacy-tree`, 73.4 min wall, 46.95 GiB
peak RSS):**

| metric | value |
|---|---:|
| blueprint ε (fleet artifact on its own tree) | **0.000248280** (reproduces the certified ≈2.5e-4; 39,508,752 info sets bit-match the census) |
| corrupted (largest subgame → uniform) | 0.008782343 (35×) |
| **repaired from boundary summary alone** | **0.000248437** (99.998% recovery, +1.6e-7 vs blueprint) |
| composed, all 495 subgames @ 120 gadget iters | 0.001696581 (expected: 120 iters can't re-attain a ~900-iter teacher ε; still 6× inside the ε=0.01 production target) |
| subgames / largest | 495; largest 1,816,272 nodes / 212,337 info sets = **0.52%** of the 349.9M-node cell |
| phase wall | build 248s · blueprint load 59s · resolve_all 3207s · repair 878s |

**The Phase-5 number:** the largest single re-solve unit is ~0.5% of the
cell. Decomposed deep-cell jobs are commodity-box-sized.

**Two production bugs found by the run and fixed en route** (each caught by
a backstop or a diagnostic, never by burning the budget):
1. O(subgames × full-tree-BR) CBV extraction — attempt 2 hit its 7h
   backstop; fixed by `resolve::boundary_summary` (one certified BR pass
   per player for the whole subgame set; values bit-identical, local
   resolve_all 32.3s→0.68s).
2. Tree-generation mismatch — attempt 3 evaluated the pre-proof-prune fleet
   artifact on today's default (7% smaller) tree; positional action-vector
   mismatch scrambled pruned decisions (blueprint certified 0.0907, the
   red flag). Fixed by running `--legacy-tree` (the artifact's own tree);
   a future improvement is action-identity projection in the CLI densify,
   as teacher export already does.

**Cost ledger (cap $5.00): $3.45 total** — retrieval $0.27, attempt 1
(harness typo, 6 min) $0.03, attempt 2 (7h backstop) $2.20, attempt 3
(63 min, mechanics validated) $0.33, attempt 4 (definitive) $0.40, disk
≈$0.22. Attempts 3+4 — the runs that produced all the science — cost $0.73
combined; the $2.20 was tuition for the O(subgames) lesson.

---

## 2026-07-21 — Phase 5 sizing scout: 0×0 trunk-region + subgame census ($0, local)

Plan 84 Phase 5 box-choice input. New `solve trunk-scout` (no arena, no
registry, no accumulators — a build-recursion replay that partitions node and
info-set counts into trunk-region vs per-subgame, mirroring
`resolve::trunk_solve`'s `n_trunk = n_full − Σ(span−1)` and its subgame
info-set partition). Totals bit-match the independent `count_tree_size` walker
(unit test + live: 8×8/30d nodes 128,394, info sets 65,410; 0×0/300d nodes
7,905,387, info sets 3,859,486 — identical to the 2026-07-17 census). Dealer 0,
Current rules; 0×0 ≡ 1×1 bit-for-bit (low scores share tree shape, no
saturation pruning).

Deal-subset sweep (0×0 tc0 d0; strided `--max-deals`; subgame count SATURATES,
trunk info sets near-saturate, everything else grows):

| deals | total nodes | total info sets | trunk nodes | trunk info sets | subgames | largest subgame (members / nodes / info sets) | RSS | wall |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 300 | 7,905,387 | 3,859,486 | 79,389 | 25,580 | 3,135 | 24 / 46,447 / 22,230 | 0.22 GB | 3.5s |
| 1,000 | 27,208,211 | 12,694,833 | 268,558 | 66,863 | 3,135 | 59 / 116,769 / 53,289 | 0.53 GB | 12.7s |
| 3,000 | 81,419,481 | 34,171,257 | 804,111 | 114,919 | 3,135 | 178 / 348,537 / 141,518 | 1.76 GB | 39.7s |
| 10,000 | 271,784,041 | 92,957,961 | 2,680,468 | 127,408 | 3,135 | 532 / 1,008,324 / 336,063 | 2.77 GB | 146s |

Full-deal (140,118) extrapolation, dealer 0, 0×0 tc0:

- **Total nodes ≈ 3.81e9** (271.78M × 14.01; additive across deals ⇒ ~exact;
  matches the plan's ~3.8B). Full info sets ~757M (sublinear, saturating —
  the local walk can't reach it: 92.96M keys already cost 2.77 GB at 10k
  deals, full set would need tens of GB, above the 24 GB box).
- **Trunk-only region ≈ 37.6M nodes** (2.68M × 14.01; additive) and **~130k
  info sets** (near-saturated at 10k). The trunk arena is **~1.0% of the full
  cell** — this is the RAM lever: a trunk-only build fits in <1 GB where the
  full 0×0 arena is ~250 GB.
- **Subgames = 3,135** (exact — saturated by 300 deals; a subgame is one
  public round-2 history, a finite set).
- **Largest subgame ≈ 7,000–8,000 members / ~14M nodes / ~2.5M info sets**
  (members ~0.053–0.06/deal linear; ~1,900 nodes/member stable; info sets
  ~deals^0.78 sublinear). That is **~0.37% of the cell's nodes** — consistent
  with the Phase-3/4 "largest re-solve unit ≈ 0.3–0.5% of cell" measurements
  (10×10 gave 1.0–1.8M nodes; 0×0's deeper live-raise ladder makes its biggest
  subgame larger). NOTE: this largest-subgame figure is a subset extrapolation,
  not a full-deal measurement — the per-subgame arena at 0×0 is the number that
  decides whether per-subgame re-solve jobs fit a commodity box, and it wants a
  full-deal walk (feasible only off-box) to pin down precisely.

Box implication: trunk-only build (~37.6M nodes, sub-GB) + on-demand
per-subgame builds (largest ~14M nodes) + compact per-subgame accumulators keep
0×0 well inside a 64–128 GB box; the ~250 GB full-arena path is what the Phase-5
infra (trunk-only + per-subgame arenas + compact accums) removes.

---

## 2026-07-27 — Plan 84 Phase 5 tail: streamed artifact, arena cache, resume-extend, cert-jobs ($0, local)

The four deferred items from the 2026-07-23 Phase-5 probe, built and gated
locally. Cell: 8×8 tc0/d0, strided deal subsets, `--rounds 6 --trunk-sweeps 1
--subgame-iters 1 --certify raw`, release build, f64 accumulators, on an M-series
laptop (24 GB). Everything below is a LOCAL measurement — the production-scale
0×0 numbers still want the $0.50 spot round-pin (see plan 84 status).

**A. Arena mode × jobs — wall and peak RSS.** `--arena-cache DIR` packs each
subgame's local arena on its first build and memory-maps it every round after;
`--keep-arenas` holds all of them in RAM; `rebuild` (the old default) replays
the engine every round.

2,000 deals (990 subgames, 89,148 trunk info sets, 3,838,352 subgame info sets):

| mode | jobs | wall s | peak RSS MB | cert s | raw eps |
|---|---:|---:|---:|---:|---:|
| rebuild | 1 | 38.68 | 590.5 | 5.3 | 0.016248627499 |
| rebuild | 4 | 18.28 | 608.9 | 1.8 | 0.016248627499 |
| **cache** | 1 | **10.57** | **593.0** | 1.9 | 0.016248627499 |
| **cache** | 4 | **6.77** | **619.0** | 0.5 | 0.016248627499 |
| keep | 1 | 7.08 | 1345.5 | 1.4 | 0.016248627499 |
| keep | 4 | 5.11 | 1353.5 | 0.4 | 0.016248627499 |

6,000 deals (990 subgames), same schedule:

| mode | jobs | wall s | peak RSS MB | cert s | raw eps |
|---|---:|---:|---:|---:|---:|
| rebuild | 1 | 114.98 | 1629.0 | 15.8 | 0.015951823535 |
| rebuild | 4 | 53.29 | 1557.1 | 5.1 | 0.015951823535 |
| **cache** | 1 | **34.43** | **1674.2** | 5.0 | 0.015951823535 |
| **cache** | 4 | **19.73** | **1742.0** | 1.6 | 0.015951823535 |
| keep | 1 | 20.19 | 3339.4 | 3.9 | 0.015951823535 |
| keep | 4 | 15.07 | 3337.3 | 1.2 | 0.015951823535 |

Reading it:

- **The arena cache buys 3.3–3.7× wall at jobs=1 (2.7× at jobs=4) for
  0.4–7% peak RSS.** `--keep-arenas` is only ~1.5× faster still and costs
  **2.0–2.3× peak RSS** — the trade the deep path cannot make at 0×0. Cache
  is now the default whenever `--checkpoint` is set.
- Disk cost of the cache: 581 MB at 2,000 deals, 1.5 GB at 6,000 — i.e. it
  tracks the arena bytes, which is exactly what the mmap keeps OUT of RSS.
- **raw eps is identical to the last digit across all six configurations** at
  both scales. Arena mode and pool size are performance knobs only.
- Init is unchanged (4.1–4.5 s / 12.1–13.4 s): the first build still happens.
  The cache pays back on rounds 2..N and on every resume.

**B. Streamed composed artifact — the 2026-07-23 post-certificate OOM.**
Same command with `--composed-out`, 2,000 deals, jobs=1, old binary (main,
d1ec412) vs new:

| | peak RSS MB | wall s | rows | file bytes |
|---|---:|---:|---:|---:|
| before (whole-profile HashMap + full-arena rebuild) | 2688.6 | 44.68 | 3,869,366 | 489,742,845 |
| after (streamed per subgame) | **618.7** | 42.63 | 3,869,366 | 489,742,845 |

**4.35× lower peak RSS, same rows, same file size**, and the artifact write
itself now costs +28 MB over the same solve without `--composed-out` (590.5 →
618.7). `compare-policies` on the two artifacts: `rows_a=rows_b=matched=
3,869,366, only_a=0, only_b=0, max TV = 0.000000, argmax_agree=1.0000` at every
depth — content-identical, only row ORDER differs (streams trunk-first then
subgame-major instead of key-sorted). The old path's memory scaled with the
whole profile AND rebuilt the full arena just to enumerate keys; at 0×0 that is
~757 M rows, which is what died after the certificate printed.

**C. Resume-extend.** A checkpoint can now raise `--rounds` (never lower it
below completed rounds). Extending a 2-round OR a 3-round checkpoint to 6
rounds reproduces a straight 6-round run's raw / tail / br / composed
certificates and game value under exact f64 equality. This required
re-anchoring the warm-up from the NEW budget and clearing the CBV tail maps at
the anchor; keeping the saved anchor instead measured raw eps 0.067 for a 3→6
extension against 0.0132 straight — the 2026-07-21 lagging-average error in a
new costume. "Solve to eps=0.01 now, extend later at zero waste" is now
literally true rather than approximately true.

**D. Cert-jobs.** `--cert-jobs` (default `min(jobs, 8)`) sizes the certificate
pool alone. Not a local win — it exists because the 0×0 certificate at
`--jobs 16` peaked at 124.5 GiB of a 128 GiB box (2026-07-23) with each BR
worker holding a whole subgame arena.

Gates: the 5 pre-existing deep equivalence tests plus 4 new ones
(`deep_arena_cache_matches_rebuild`, `deep_streamed_artifact_matches_in_memory_profile`,
`deep_resume_extend_matches_straight_run`,
`deep_resume_cannot_shrink_below_completed_rounds`). `make check` green.

Cost: $0 — entirely local, no cloud, no full-deal builds.
