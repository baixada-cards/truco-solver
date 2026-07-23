# Plan 83 — Neural policy approach

Owner decision 2026-07-17: after the exact-solving cost program landed at
~$12–16K for the whole grid (still >10× the <$1K goal, walled by deep symmetric
cells — see [EXACT_SOLVING.md](../EXACT_SOLVING.md)), invest the limited budget
in a neural policy instead of buying more exact structure. This supersedes the
framing of `SOLVER_PLAN.md §14` ("solve fully, then distill"): we do NOT have a
full tabular solution and will not pay for one. We have a partial exact corpus
and will learn from it, then extend.

## The strategy (as stated by the owner)

1. **Supervised learning (SL) first**, on the spots we have exact solutions for.
   Fit a network to reproduce the equilibrium action distributions.
2. **Then decide how to continue.** Expectations set up front:
   - SL will likely reproduce the solved region (esp. 10×10) very well. It is a
     clean supervised target and SL is far easier than deep RL, so **SL quality
     is a practical upper bound on the easy path** and a strong sanity check.
   - **Deep RL** (self-play) is the route to the *unsolved* deep bands, and it
     should benefit from **warm-starting** on an SL net that already plays the
     solved region well.

## Corrected training-data inventory (read before building the dataset)

The exact corpus is smaller than the shipped Study release implies. Genuine
from-scratch exact solves (EXACT_SOLVING.md §3):

- **10×10** — 9 turn-up classes, dealer 0 (dealer 1 by the symmetry
  `mv(a,b,d)=1−mv(b,a,1−d)`), ε≈2.5e-4. ~39.5M info sets per (TC). This is the
  bulk of the SL data and the one {1,3}-band exemplar.
- **Mão-de-onze row 11×0 … 11×11** — all 12 scores, all 9 TCs, both dealers
  (`teacher2-20260704`). Smaller trees (~5.6M info sets at the mão band). These
  are structurally special (accept/fold-eleven decision).
- **11×10, 11×9** — dedicated tight solves, both dealers (11×10).

NOT exact (do not train on as ground truth): 10×9, 9×9, and everything with
`min(score) ≤ 8`; and 220 of the 225 shipped Study profiles (transfers). Pull SL
targets from the genuine solve buckets, not the release.

Target representation decision to make explicitly: **raw CFR average** (the true
equilibrium mix) vs **purified average** (dominated actions re-zeroed,
`teacher_export.rs`). Raw is the faithful target; purified is what ships. Prefer
raw for SL unless there is a reason otherwise.

## Assessment (Claude, 2026-07-17)

**What SL will and won't buy.**
- It will fit 10×10 and the mão row well — clean targets, and the measured
  structure is favorable: 52–56% of on-path play is near-pure (easy to nail),
  and turn-up class barely changes on-path play (RN 29, `policy-stats` /
  `compare-policies`), so a net conditioned on (hand, history, score, tc,
  dealer) can share heavily across TC. Reach-weighted, ~45% of play is genuinely
  mixed — that is the hard residual SL must actually fit, not memorize.
- It will NOT, by itself, tell us how to play the **unsolved deep bands**
  (0×0 … 8×8 region). Those trees are deeper (longer raise ladders) and
  strategically different; a net trained only on {1,3}+mão has never seen a
  live retruco/vale-nove decision. **The open empirical question is
  generalization**: does an architecture conditioned on score+ladder-depth
  extrapolate sensibly to deeper ladders, or does it need RL to ever see them?
  This is the crux and should be measured early (train on 10×10, evaluate
  best-response exploitability on a held-out deeper band — expect it to be poor,
  and quantify how poor).

**Architecture implications from what we already measured.**
- Condition on (score, tc, dealer) as inputs so one net covers the lattice. The
  similarity results say: TC can be a low-weight input (near-invariant on path),
  score must be a real input (on-path play is score-sensitive, esp. shallow
  depths — the mão accept/fold and first plays). See RN 29 depth breakdown.
- The `min(score)` band determines the legal action set (which raises exist).
  Feed ladder-depth / legal-action mask explicitly; do not make the net
  rediscover the pruning rules.
- Card features as strength vectors (manilha rank, plain rank), history as an
  action-type sequence. MLP is likely enough for a first pass; a small
  transformer over history if needed (SOLVER_PLAN §14 architecture notes).

**Why SL-as-warm-start-for-RL is the right sequencing.** Deep RL from scratch on
a bluffing imperfect-information game is expensive and unstable; the poker
literature (DeepStack, ReBeL, Deep CFR, NFSP) warm-starts or anchors on solved
subgames / supervised targets. Our exact corpus is a free, high-quality anchor
for exactly the states RL would otherwise waste compute rediscovering. Concrete
risk to watch: RL drifting the net off the SL-correct region while chasing deep
bands (catastrophic forgetting) — anchor with a distillation/regularization loss
toward the exact targets on the solved region.

**Evaluation must reuse the exact tooling.** The decisive metric is
**exploitability of the net**, computable exactly with `compact-br` against the
net's policy (arena-free, fits 16-GiB workers even at 0×0). This is the honest
"is the net actually good" gate — head-to-head vs tabular and KL-to-tabular are
secondary. Any RL claim about a deep band must be certified this way, on the
tree it plays, with a complete match-value table (the compact-br guardrail
enforces the latter).

## Concrete first steps (not yet executed — "then we decide")

1. **Dataset export.** A `solve export-nn` (or reuse `.teach`) pass emitting
   flat `(info-set features → action-probability vector)` rows from the exact
   buckets. Decide raw vs purified. Start with 10×10 (all 9 TC, d0) + mão row.
2. **SL baseline.** MLP conditioned on (features, score, tc, dealer, legal
   mask); cross-entropy / KL to the target mix. Measure KL-to-tabular AND
   exact exploitability of the net on 10×10.
3. **Generalization probe.** Train on 10×10 only; certify exploitability on a
   held-out deeper band (e.g. one {1,3,6} spot). Quantify the SL ceiling on
   unseen ladder depth — this decides whether RL is required or merely helpful.
4. **Then decide** on the RL phase (self-play / Deep-CFR-style), warm-started
   from the SL net, anchored to the exact region.

Tooling note: `autoresearch/` (Karpathy-style harness around
`cfr_experiment.rs`) is the intended experimentation surface; the NN work will
likely need its own training loop, but reuse the solver for data export and for
exact exploitability evaluation.

## Open questions

- Does one net conditioned on score/ladder-depth generalize across bands, or do
  we need per-band nets / curriculum? (Step 3 answers the SL half.)
- Raw vs purified targets — does training on purified hurt the mixed 45%?
- What exploitability can SL alone reach on 10×10, and how far does it degrade
  one band deeper? (The upper-bound claim, quantified.)
- RL algorithm choice and the anchoring/regularization scheme to prevent
  forgetting the exact region.
