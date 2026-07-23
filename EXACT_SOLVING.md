# Exact Solving — Consolidated Status & Resume Guide

**Status: PAUSED 2026-07-17.** The program to solve (more of) Truco exactly is
deliberately parked in favor of a neural-policy approach (see
[plans/83-neural-policy-approach.md](plans/83-neural-policy-approach.md)). This
document is the single authoritative map of what was learned, so the exact
line can be resumed cold. It summarizes and cross-references the detailed
records; when a number here disagrees with `SOLVER_BENCHMARKS.md`, that file
(dated entries) is authoritative.

Primary detailed sources: `SOLVER_BENCHMARKS.md` (every measured run, dated),
`RESEARCH_NARRATIVE.md` (numbered narrative items, the "why"),
`plans/79-solver-cost-optimization-program.md` (the optimization program),
`SOLVER_PLAN.md` (architecture + §13 cost, §14 distillation).

---

## 1. Why we paused

The stated goal was a useful full-game policy for **<$1,000** (stretch
**<$100**). The honest landing after the 2026-07 optimization program:

- **~$31K** to solve the whole grid at ε=0.01 with warm starts (evidence-based).
- **~$12–16K** after asymmetric raise pruning (2026-07-17, the last win).
- Still **~12–16× above the goal**, and the residual is structural: a handful
  of deep symmetric cells stay huge and no lossless trick shrinks them.

The two levers that actually moved cost were **accepting approximation**
(ε=0.01) and **warm starts** — "solve less," not "solve cheaper per unit." The
week of memory-representation engineering was lossless and correct but did not
move the headline, because the deep bands are structurally large. The
conclusion (RESEARCH_NARRATIVE items 31–32): the remaining gap, if it closes,
closes through **structure** — dominance, abstraction, or a cheaper deep-band
algorithm — not more tuning. Rather than keep buying structure at diminishing
returns, spend the limited budget on the neural approach.

The intent was never necessarily the *full* game — it was to solve **as much as
possible exactly** within budget. This document therefore also records exactly
where the "affordable exact frontier" sits.

---

## 2. The core mental model: cost is governed by `min(score)`

The single most important fact. The raise-prune rule
(`game_tree.rs:abstract_legal_actions`) removes raises once
`min(score) + on_table_stake >= MATCH_TARGET(12)`. So the game tree — and thus
solve cost — depends **only on which of the raise ladder is still live**, which
depends only on `min(score)`. Verified exactly: 9×9 = 10×9 = 10×10; 6×6 = 8×8;
3×3 = 5×5; 1×1 = 2×2 = 0×0.

| tier (live ladder) | `min(score)` | info sets (all deals, per TC) | nodes | example scores |
|---|---|---:|---:|---|
| mão de onze | 11 | 5,611,123 | 49.6M | 11×11 |
| {1,3} | 9–10 | 39,508,752 | 352.3M | 10×10, 9×9, 10×9 |
| {1,3,6} | 6–8 | 129,144,643 | 1.11B | 8×8, 9×6, 6×6 |
| {1,3,6,9} | 3–5 | 341,656,035 | 2.87B | 5×5, 9×3 |
| {1,3,6,9,12} full | 0–2 | 812,865,845 | 6.70B | 0×0, 9×0, 1×1 |

**Job counts** (ordered `(a,b)` pairs, `a,b∈[0,10]`, one solve per pair by
dealer symmetry `mv(a,b,d)=1−mv(b,a,1−d)`, ×9 TCs): {1,3}=36, {1,3,6}=189,
{1,3,6,9}=351, full=513. The full-ladder tier is ~89% of grid cost.

**Asymmetric refinement (2026-07-17, accepted):** the prune argument works
per-acting-player, not just on `min`. Gating on the *acting* player's own score
(`TreeRules::AsymmetricRaisePrune`) shrinks lopsided cells: 9×6→0.66×,
9×3→0.27×, 9×0→0.11×; symmetric cells unchanged. This is what dropped the
bracket to ~$12–16K. See §5 and `SOLVER_BENCHMARKS.md` 2026-07-17.

---

## 3. What is actually solved exactly (the SL dataset boundary)

Genuine from-scratch CFR solves at tight ε (not transfers). This is the exact
region — everything else in the shipped Study release is transferred/certified,
NOT exact.

| region | run / GCS bucket | coverage | ε (raw) |
|---|---|---|---|
| **10×10** | `10x10-full-20260709/d0/` | 9 TCs, dealer 0 (d1 by symmetry) | ~2.5e-4 |
| **mão row 11×0 … 11×11** | `teacher2-20260704/solutions/` | all 12 scores, all 9 TCs, both dealers | tight (teacher-grade) |
| **11×10** | `11x10-20260702/{d0,d1}/` | 9 TCs, both dealers | tight |
| **11×9** | `11x9-20260703/d0/` | at least tc0 d0 | tight |
| refined tremble variants | `tremble-refine-20260711/`, `br-gap-20260711/` | 5 shipped spots, off-path repaired | tight |

**Not exact:** 10×9, 9×9, and everything with `min(score) ≤ 8`. The 225-profile
Study release (`profile-transfer-fleet-20260716`) is **5 exact checkpoints +
220 transfers** — do not treat it as an exact corpus.

So the affordable exact frontier reached is essentially **10×10 + the entire
mão-de-onze row**. In cost terms (§2), that is the two cheapest tiers; the
{1,3,6} tier and deeper were never solved.

Artifacts are purified average strategies (`teacher_export.rs` purification at
negligible-mass actions). For an exact *target* (e.g. SL), decide raw-average
vs purified deliberately — purification re-zeros dominated actions and is not
the literal CFR average.

---

## 4. Cost brackets (how the number evolved)

All spot pricing, N2 custom-extended, single-threaded solve (cost ∝ RAM-hours,
CPU count only matters for the minutes-long build). Full derivation:
`SOLVER_BENCHMARKS.md` "Refined cost estimate" + 2026-07-17 entries.

| stage | whole-grid spot cost | source |
|---|---:|---|
| raw grid, tight ε≈2.5e-4 | ~$505K | baseline table |
| ε=0.01 relaxation | ~$50.5K | Phase 3 |
| + same-band warm starts (1.61×) | **~$31K** | Phase 2, evidence-based anchor |
| + asymmetric raise pruning (0.38–0.53×) | **~$12–16K** | 2026-07-17 |

Separately, **exact whole-grid certification** (not solving) of an
already-solved policy: ~$10K materialized → **~$8K** with compact BR, which now
fits 16-GiB workers even at the deepest band (Phase 8; compact BR measured
5.95 GiB at 10×10, 8.13 GiB at 0×0). Certification is cheap relative to solving.

Known unquantified risk carried throughout: **iteration counts** for the
untested deep tiers are assumed ~equal to the 10×10 anchor (~900 tight / ~40–90
warm). Deeper raise ladders may need *more* iterations, not just bigger trees.
Only tree size (not convergence rate) was measured for {1,3,6} and deeper.

---

## 5. Optimization ledger (every lever, its verdict)

Status key: ✅ accepted/deployed · 🧪 implemented, gated, not fleet-credited ·
❌ rejected · ➖ measured negative/no-op.

| lever | verdict | measured effect | where |
|---|---|---|---|
| ε=0.01 relaxation | ✅ | $505K→$50.5K; ≤0.5–1pp equity | Phase 3 |
| Same-band regret warm starts | ✅ | 1.61× wall (10×10→10×9) | Phase 2; RN item ~ |
| Cross-turn-up / cross-score transfer | ✅ (portfolio) | cheap where it certifies; non-monotonic — see canaries | RN 26; fleet |
| Proof-scoped hidden-play + forced-fold prunes | ✅ | 1.76–2.26× raw nodes; ~7% info sets; no cost credit | Phase 7; RN 15 |
| **Asymmetric (per-player) raise pruning** | ✅ | **grid 0.38–0.53×; full tier 0.36–0.51×; $31K→~$12–16K** | 2026-07-17; RN 31 |
| SyncCFR+ (vs async) | ✅ | ~10× iteration-count reduction, same ε | production |
| Compact exact-BR certification | ✅ | arena-free; 5.95 GiB @10×10, 8.13 GiB @0×0; ~$10K→$8K | Phase 8 |
| Dense-direct artifact writes (phase 9c) | ✅ | shallow warm peak 31.6→16.2 GiB; class chain 75→55→40→24 GiB | Phase 9; RN 28 |
| Direct-to-dense + row-streamed warm load (9a/9b) | ✅ | 61.5→43.6→31.6 GiB peak | Phase 9 |
| Legacy-tree eval + projection guardrails | ✅ | old checkpoints certifiable on own tree; fail-fast on bad projection/mv | Phase 8; RN 27–28 |
| Restricted solver + exact-BR oracle | 🧪 | safe but 49% slower than full warm; no strict-target credit | Phase 2b |
| Reversible regret pruning | 🧪 | accuracy-safe, zero stopping-time benefit at ε=0.01 | Phase 4 |
| Reduced-precision (`f32`) accumulators | ⏳ | not yet A/B'd | Phase 6 (open) |
| Deep-band mini-batched MCCFR | ❌ | held-out ε worsened 0.226→0.241; do not scale | Phase 5 |
| Static all-action BR-union restricted arena | ❌ | closure not materially < raw tree | Phase 1 |
| Observable-card forced-fold generalization | ➖ | zero structural change (proof-level closure) | Phase 7 |
| Ex-ante hand-strength/equity dominance | ❌ | bluff raises make low equity strategy-dependent | Phase 7 |

---

## 6. Open problems / what could still move the exact number

If resuming, these are the live threads, roughly by promise:

1. **Deep symmetric cells are the residual wall.** After asymmetric pruning,
   ~9 cells (0×0, 1×1, 2×2, 0×1, 0×2, 1×2 and mirrors) do NOT shrink and still
   need a >150 GiB worker (measured: the 0×0 solve build+start exceeded 148.5
   GiB and livelocked a 160 GiB box — `SOLVER_BENCHMARKS.md` 2026-07-17, a
   negative result). A real deep-solve needs a ≥256 GB worker **and** a
   fail-fast RSS guard. This is the single most expensive measurement in the
   program and was never completed.
2. **Iteration-count for deep tiers is unmeasured** (§4 risk). One ≤$2 scout
   solving a single {1,3,6} spot to ε=0.01 would convert the deep-tier cost
   from projection to measurement.
3. **More asymmetric-style structural prunes?** The per-player idea (RN 31)
   came from noticing a game-theoretic redundancy in the tree. There may be
   others (e.g. score-dependent bluff-range collapses). Each needs the Phase-7
   discipline: strategy-independent payoff proof → small-tier exact-BR A/B →
   count-tree. **2026-07-17 audit — two candidates closed empty:**
   - *Residual card/suit isomorphism*: ~1.0×, nothing left. Resolved by
     inspection: `InfoSet` (`info_set.rs:93`) is already the canonical
     quotient — plain cards are suit-independent (`AbstractCard::Plain`),
     manilhas totally ordered, starting hand stored sorted, and
     `player`/`is_dealer`/`turnup_class`/`history` are each strategically
     necessary (collapsing `is_dealer` was the 11×10 exploitability wall).
   - *Per-node match-saturation / forced-winner collapse*: ~0 on the cells
     that matter. The stake-saturation form is exactly the deployed
     `min+stake>=12` raise prune; the forced-winner-by-card-strength form is
     only lossless at frozen stake (live raises keep bluff/betting value
     strategy-dependent), and card play is <1% of a full-ladder cell
     (count-tree, 300-deal subset: 11×11 = 45,723 nodes vs 0×0 = 7,905,387;
     full-deal docs ratio 49.6M/6.70B ≈ 0.74%). Same runs confirmed the
     symmetric wall: asymmetric pruning leaves 0×0/1×1 counts bit-identical.
   - *Meta-result*: >99% of the expensive symmetric cells is live
     betting/bluffing tree, which is strategy-dependent and therefore immune
     to the whole lossless-pruning family. Remaining exact-solve savings live
     in decomposition (CFR-D / safe subgame re-solving — thread 6 below) or
     representation (threads 1, 4), not in further pruning.
4. **Reduced precision** (Phase 6) is the one remaining un-run lossless-ish
   representation A/B.
5. **Lossy abstraction** — the only lever with order-of-magnitude potential and
   the one deliberately never taken: coarsen the hand abstraction or the score
   lattice, trading exactness for size. This is where a genuine <$1K path would
   have to come from, and it is a research bet, not a tuning pass.
6. **Safe subgame re-solving (CFR-D / continual re-solving)** — the "cheaper
   deep-band algorithm" candidate, and after the 2026-07-17 prune audit (thread
   3) the only lossless-to-ε lever left that structurally attacks the symmetric
   deep cells: solve a trunk, then re-solve each subgame independently behind a
   safe opt-out gadget (Burch–Johanson–Bowling 2014), never materializing the
   whole 1.54 TB tree. Exploitability provably bounded by the blueprint's.
   **BUILT AND VALIDATED AT PRODUCTION SCALE (2026-07-18, plan 84 Phase 3):**
   round-2 boundary decomposition + BJB gadget over the existing packed trees
   (`subgame.rs`, `resolve.rs`, `cfr::resolve_subgame`, `solve
   resolve-subgames`). At full 10×10 tc0/d0 against the real fleet blueprint:
   a subgame corrupted to uniform (ε 0.000248→0.00878) was repaired from the
   boundary summary ALONE back to ε 0.000248437 (+1.6e-7); the largest of 495
   re-solve units is 0.52% of the cell — deep-cell jobs become
   commodity-box-sized. Total validation cost $3.45. **FROM-SCRATCH LOOP
   VALIDATED (2026-07-21, plan 84 Phase 4):** `solve trunk-solve` solves
   10×10 with no blueprint anywhere — raw certificate 0.001122, game value
   within 5.5e-5 of the teacher ground truth, ~1.8× monolithic wall net.
   The one production lesson: intra-round couplings must read the CURRENT
   regret-matching iterate, not the lagging average (a ~0.018 plateau
   otherwise; three-certificate diagnostic + fix in `SOLVER_BENCHMARKS.md`
   2026-07-21). 8 runs, $6.70 total across Phases 3–4. Remaining for the
   deep-band payoff: the composed deep-cell benchmark on commodity boxes
   (Phase 5 — trunk-only arenas + per-subgame remapping + fleet shape).

---

## 7. Resume guide: tooling & operational patterns

### Solver CLI (`crates/truco-solver/src/bin/solve.rs`, `cargo build --release --bin solve`)
- `solve-tc --score SxS --tc N --dealer D [--eps E] [--max-iters K] [--algo sync] [--warmstart-from CKPT] [--warmstart-profile-transfer] [--asymmetric-raise-prune] [--max-deals N] --match-values MV --checkpoint CKPT --data-dir DIR` — the solve. Writes strategy + `.gv` + full checkpoint. Streams artifacts from dense accumulators (phase 9c).
- `count-tree --score SxS --tc N [--asymmetric-raise-prune] [--legacy-tree] [--max-deals N]` — tree size without solving; the cheap ($0 local) structural instrument.
- `compact-br --policy-checkpoint CKPT --match-values MV --score SxS [--legacy-tree] [--project-dominated remap|renormalize] [--control]` — arena-free exact BR / certification. Aborts if any non-terminal successor mv cell is unsolved.
- `compare-policies --a A.bin --b B.bin [--remap-turnup] [--reach-weighted] [--legacy-tree]` — descriptive similarity of two solved policies (RN 29).
- `policy-stats --strategy S.bin [--reach-weighted]` — purity/mixing distribution.
- `allocation-scout` — sampled reach/error prioritization (Phase 8), not a certificate.
- `set-mv --mv-out F --set s0:s1:d:v ...` — build/edit a match-value table (used synthetic complete tables for the asymmetric value A/B).

### Tree rules (`game_tree.rs:TreeRules`)
`Current` (default) · `LegacyPreProofPrunes` (pre-2026-07-16 tree, for evaluating old artifacts on their own tree) · `AsymmetricRaisePrune` (Current + per-player raise prune). Threaded through `count_tree_size_with_rules`, `build_all_trees_with_dealer_rules`, `SolveConfig::tree_rules`, and compact BR.

### GCS layout
`gs://truco-solver-runs/<run>/…`: `solutions/<score>/tcN.dD.bin` (purified avg strategy), `.ckpt.bin` (full resumable checkpoint), `.gv` (game value sidecar), `treecache/*.trees` (band-shared arenas), `match_values.bin`. Cost-opt scouts under `cost-opt-2026071{6,7}/`.

**Storage: the bucket is ARCHIVE-class as of 2026-07-17** (lifecycle rule:
transition to ARCHIVE at age 1 day). 674 GiB dormant, ~$0.81/mo at Archive vs
~$13.5/mo at Standard. Consequence for resume: reads incur a retrieval fee
(~$0.05/GB, e.g. ~$8 to pull the ~156 GiB of exact strategy `.bin`s for SL),
and Archive has a 365-day minimum-storage-duration early-deletion fee (tiny at
Archive rates). The exact-solve outputs dominate the footprint — teacher2 alone
is 363 GiB (149 GiB strategies + 186 GiB full checkpoints + 28 GiB treecache).
Checkpoints are only needed to resume/warm-start; treecaches are
deterministically rebuildable from code (minutes). If a hard cleanup is ever
wanted, those two categories plus the superseded `10x10-20260703` run are the
safe targets; keep every strategy `.bin` and `.gv`.

### VM pattern (memory: `project-gcp-solver-export-vm-pattern`)
Spot N2 custom-extended, `--instance-termination-action=DELETE`,
`--max-run-duration` backstop, startup-script that builds from a tiny source
tarball (`Cargo.toml`+`Cargo.lock`+`crates/`), pulls checkpoint+mv, runs, uploads
incrementally with a DONE/FAILED marker, self-deletes. **Size the worker for the
MEASURED peak with headroom, and add a fail-fast RSS guard** — the 2026-07-17
deep-solve livelock was an under-provisioning + no-swap silent-thrash failure.
Watch `SSD_TOTAL_GB` (separate quota; pd-balanced counts) and
`CPUS_ALL_REGIONS` (32 default). Local constraint: this workstation has
<11 GiB free disk and cannot hold a full-deal tree — never run full-deal solves
locally (memory `feedback_memory_usage`).

---

## 8. One-paragraph handoff

We can solve the two cheapest tiers exactly (10×10 + the mão row, done) and,
with all optimizations, the whole grid for ~$12–16K — still >10× the budget,
walled by a few deep symmetric cells that no lossless method shrinks. The
tooling to solve, certify, count, and compare any spot is built and documented
above. The exact line is paused, not abandoned: if a cheap deep-tier
iteration-count scout or a new structural dominance rule lands, the bracket
could move again. But the decision (2026-07-17) is to invest the remaining
budget in a neural policy — see plans/83.
