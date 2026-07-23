# Plan 71 — Neural distillation pipeline (supersedes the full tabular grid)

Decision (2026-07-04): the full tabular solve is priced out (~$5-12k even after
the optimization campaign — see SOLVER_BENCHMARKS 2026-07-03/04). The product
of this project becomes a **neural net trained on exact subgame solutions**,
with a measured exploitability number. Budget target: **O($100)** for steps
1–3; step 4's scope is deliberately deferred (re-estimate after 1–3).

## The four steps

### 1. Teacher subgames — solve a diverse band sample to very low ε (~$40–80)

Post-optimization prices (sync + pruning + packed/SoA + band-shared treepacks)
make "a few subgames" ≈ 40 subgame-jobs across three structurally different
bands:

| Band | states | ε target | est. cost |
|---|---|---|---|
| mão de onze (11-column, finish it) | (11,0..9) + done ones | 0.001 | ~$10 |
| {1,3} triangle (10×10, 10×9, 9×9) | 3 states | 0.001 | ~$5–10 |
| {1,3,6} sample (e.g. 8×8, 8×6) | 2–3 states, subset of TCs ok | 0.003 | ~$25–50 |

All of it feeds both the statistics (step 1a: accept ranges, bluff
frequencies, position effects, truco-call thresholds by band) and the training
set. More teacher data is nearly free — take breadth over depth of ε.

**Execution status (2026-07-04, staged):**
- Stage A (RUNNING, `truco-teacher2`, n2-highmem-16): the full 11-column at
  ε=1e-5 — 11×11's remaining 17 (tc, dealer) jobs, then rungs (11,10)→(11,0),
  18 jobs each, aggregating mv(11,s,·)+mirrors between rungs via `.gv`
  sidecars + `set-mv` (recipe: `tools/gcp/startup-teacher2-column.sh`).
  Benchmarks moved the target from the table's 1e-3 to 1e-5: 11×11 tc0 d0
  reached ε=1.0e-5 in 78 min on one core (~$0.5).
- Stage B (queued): {1,3} triangle at ~1e-4, priced off the 10×10 tc0 d0
  benchmark still running on `truco-teacher1`.
- Stage C (queued): {1,3,6} sample (8×8, 8×6), subset of TCs.

Engineering item: **compact strategy export** — SHIPPED 2026-07-04 as
`export-teacher` / `teacher_export.rs`. Per-state `.teach` files carry, in
`table_idx` order: f32 strategy probs, **per-action one-shot-deviation Q
values** (the "EV left on the table" numbers — added for the EV-feedback
product feature and as step-2 Q-head training targets), and own/opponent
counterfactual reach masses for importance sampling. Info-set metadata lives
once per band (`--band-meta` sidecar or the treepack; same `sig_hash`
binding). Measured on the real 11×9 tc0 d0 solve: 709 MB `.bin` → 152 MB
`.teach` → **28 MB** zstd-3; the Q pass is exact (root value reproduces the
solve's `.gv` to 2e-11) and takes 1 s for 5.6M info sets.

### 2. Supervised distillation (~$0–15)

(info-set features → action distribution) pairs from the step-1 solutions.
Small MLP (~100k params, phone-deployable — the SOLVER_PLAN §14 goal).
Feature design constraint (load-bearing, see step 3): **score and current
stake are INPUT FEATURES, not separate heads**, and the output layer covers
the FULL action space with legality masking.

### 3. Self-play extension to the full game (~$20–50)

Initialize from the step-2 net. Because score/stake are inputs and the action
space is masked-full, the net enters uncovered states (min(score) ≤ 5) already
playing its learned {1,3,6}-band strategy — a far warmer start than random:
at 0×0 it simply never re-raises past 6 at first, and self-play only has to
learn *when the 9/12 rungs beat the strategy it already has*, not how to play
truco. Use NFSP-style anchoring (or best-response pools) rather than naive
self-play — raw self-play in imperfect-info games can cycle.

This is the quality-risk step, not the cost-risk step. Checkpoints of the net
are cheap; evaluate against the step-1 exact strategies (KL / action-match on
held-out solved states) continuously.

### 4. Full-game exploitability of the net — SCOPE DEFERRED

Details to be decided after steps 1–3 land; re-estimate price then. The two
candidate designs, for the record:
- **Two-tier (~$60)**: exact BR where trees are cheap (the teacher bands —
  a true bound there) + sampled exact BR on a few streamed full-ladder
  (state, TC) pairs (a measured estimate for the deep band).
- **Full exact certification (~$300–500, one-time)**: the backward-induction
  BR sweep over the whole (score, dealer) DAG — the paper-grade headline
  number. Available later if the two-tier estimate looks good.

Mechanically: BR-against-the-NN = batched NN inference over a subgame's info
sets (~minutes on a GPU) filling a `DenseAccum`, then the EXISTING exact-BR
code runs unchanged. The treepack cache means certifying net v1, v2, … pays
only BR passes.

## Storage policy (the $/month question)

- **Do not store cheap-band treepacks.** Mão rebuilds in ~1 min, {1,3} in
  ~7 min — storage would cost more than rebuilding. Cache only the {1,3,6}
  artifacts actually used in step 1 (2–3 × ~22 GB, Nearline).
- Solutions in the compact strategy-only format + zstd.
- Bucket lifecycle: >30 d → Nearline, >120 d → Coldline.
- Current bucket: 112 GiB ≈ $2.3/mo; projected through step 3: **≤ $5/mo**
  (nothing like the $50–100/mo full-tree-cache scenario, which is cancelled).

## What this supersedes / keeps

- Supersedes: the full-grid tabular descent (SOLVER_PLAN pipeline milestones).
  The lattice work above min(score) ≥ 9 stands and is reused as teacher data.
- Keeps: exact per-info-set BR as the measurement instrument; treepack/mmap
  infra (serves steps 1 and 4); the full-game certification sweep design
  (now step 4's "full" variant).
- Parked explicitly: {1,3,6,9} and full-ladder tabular solves; implicit
  betting-dimension factoring (plan 70 addendum) — only revisit if step 4's
  full variant is commissioned and streaming underdelivers.
