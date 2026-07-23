# Questions about optimal play

Open questions about what equilibrium truco actually looks like, collected while
studying the solved charts. Each entry states the question, why it is
interesting, and what would settle it. Resolve entries in place (keep the
question, add the answer and the evidence) rather than deleting them.

Method note: "the solve" below means the certified exported strategies read
through the study lab. Off-equilibrium infosets carry untrained mixes (see
RESEARCH_NARRATIVE.md, open question 0) — any investigation must restrict
itself to holdings the lab does not flag as weakly trained.

## Q1 — Does equilibrium play ever hide a card? (2026-07-10, open)

Hiding (playing face down) always loses the trick, so it can only pay as
information denial: giving up a trick you were losing anyway without showing
what you hold. The suspicion is that genuine hides exist but are rare, mostly
as trick-1/trick-2 replies from hands that cannot win the trick, and that a
large share of the hide mass visible in charts today is either (a) untrained
off-path noise, or (b) EV-ties where hiding and openly discarding are exactly
equivalent (against a card that beats everything you hold, every action loses
the same).

To settle it: sweep all certified spots for infosets where a hide action has
p above the display-suppression threshold, self-loss within the trained
tolerance, AND a strictly positive q-gap over the best open play — i.e. hides
the solve prefers, not merely tolerates. Catalogue them by round, role, and
stake. The study lab's Cost view ships with hides excluded by default until
this is understood.

Evidence so far (2026-07-10): `11x10 v4 : a 5d 3 / q`, pé answering the Q
lead holding 6-4 (full hand 3-6-4). All four actions — play 6, hide 6, play
4, hide 4 — export identical q = −1.000 and pts = −3.0: the hand is dead
whatever pé does, mão's round-3 play is forced so information has no value,
and CFR leaves the exactly-uniform 25/25/25/25 mix. A trained, on-path,
unflagged hide that is pure indifference — category (b) as suspected. No
strictly-preferred hide has been observed yet.

## Q2 — Do off-path garbage mixes matter for distillation? (2026-07-10, open)

We plan to distill the solved strategies into neural networks. Untrained
off-path infosets export garbage mixes (RESEARCH_NARRATIVE.md, open question
0). If training samples are drawn reach-weighted, garbage gets ~zero weight
and should not hurt; if the net is trained on uniformly sampled infosets, or
evaluated off-path, garbage targets inject noise exactly where the net has to
generalize. Options if it matters: reach-weighted sampling, dropping
high-self-loss infosets from the training set, or re-exporting with q-argmax
purification at negligible-reach infosets (which changes the targets from
"equilibrium" to "equilibrium where trained, greedy elsewhere" — defensible,
but a modelling choice that must be made deliberately, not by accident).

## Q3 — Should we train off-equilibrium spots, and how? (2026-07-10, ANSWERED 2026-07-11)

Untrained infosets matter twice: as distillation targets (Q2) and because a
deployed bot can be steered into them by humans playing "wrong". Three
routes, roughly cheap to gold:

1. **Perturbed-game solving (trembling hands).** Add ε-exploration to the
   solver traversal so every infoset accumulates visits; the averaged
   strategy is then defined everywhere and, as ε shrinks, approximates a
   sequentially rational refinement (quasi-perfect flavour). One solver
   change, applies to all future fleet solves, perturbs on-path targets by
   O(ε). The cleanest single fix.
2. **Post-hoc q-argmax at flagged infosets.** Free with current exports and
   the study lab's untrained flag (self-loss OR own-reach). Caveat: q at
   deeply off-path nodes is evaluated against garbage continuations, so the
   argmax is only trustworthy within about one step of the trained tree —
   good as a bot-side guard (never hide a winner), not as ground truth.
3. **Subgame resolving on demand.** The gold standard for play. The earlier
   RAM objection applies to full-game solves; post-round-1 subgames here are
   tiny (≤403×403 ranges, ≤3 card turns plus the raise ladder) and would
   resolve in milliseconds, even in WASM. Safe resolving needs boundary
   values, which the exports already carry (q).

Decision (2026-07-10): route (1) goes first, as warm-started refinement of
the existing checkpoints — the Phase-5c full-state `--resume` makes this a
budgeted top-up (~10–25 USD across 11x11 d0/d1, 11x10 d0/d1, 10x10 d0), not
a re-solve. Acceptance: the study lab's untrained-infoset share collapses and
the known garbage probes (e.g. `11x10 v4 : a 4 4` hiding winners) get sane
mixes, with re-certified eps. Route (2) stays as the bot-side guard, route
(3) — a WASM resolver, including fix-one-side maximally-exploitative
counter-strategies and (with a warning) full round-1 at 0×0 — is deferred
past v0; a best response against a FIXED strategy is a single traversal, no
iterations, so it is compute-light when we get to it.

**Outcome (2026-07-11):** route (1) shipped. `cfr::TrembleSchedule` floors
every player node's strategy at `σ'(a) = ε/|A| + (1-ε)·σ(a)` before it is
used for reach propagation, regret, AND average-strategy accumulation
(`--tremble-eps`/`--tremble-eps-end` on `solve-tc`, annealed 0.05→0.01, flag
default off). All 5 target spots (11×11 d0/d1, 11×10 d0/d1, 10×10 d0, tc0 —
the columns actually shipped) were warm-started from their existing
checkpoints with +200 iterations (11-column) / +100 iterations (10×10, its
tree is ~7× bigger and the tremble floor defeats the zero-prob pruning fast
path, so iterations cost ~5s each at 11×11/11×10 scale vs ~85s each at
10×10 scale — full numbers, cost, and the RAW-vs-PURIFIED own-reach nuance
below are in `SOLVER_BENCHMARKS.md` 2026-07-11 and `RESEARCH_NARRATIVE.md`
2026-07-11).

The concrete garbage probe from `RESEARCH_NARRATIVE.md` open question 0
(`11×10 d1`, history `[33,0,0]`, mão holding 5♥5♠+4, reach ~6.4e-6) is fixed:
before, the two manilha HIDE actions (certain loss, q=−1.0) each carried
p≈0.21 versus p≈0.29 for the correct PLAY; after, the raw average puts
98.3% combined mass on the two PLAY actions and purification zeros the HIDE
actions outright (p=0.0). Aggregate self-loss-flagged share (self-loss >
`assert_qgap_pp`) collapsed 8–16× across the 4 cheap spots and ~1.9× at
10×10 (which only got half the added iterations at 7× the tree size).
Own-reach did NOT collapse to ~0 when measured on the shipped, PURIFIED `p`
— that's expected, not a bug: purification correctly re-zeros genuinely
dominated actions after averaging, so descendants of a correctly-pruned
branch keep near-zero purified reach regardless of how well-trained they are
internally. Measured on the RAW (pre-purification) average strategy instead,
own-reach-flagged share collapsed to exactly 0% at 11×11 and dropped ~5× at
11×10 (deeper histories mean the `(ε/|A|)^depth` floor itself drops under
the 1e-3 threshold at depth 4, which is expected math, not a defect) —
confirming the mechanism does exactly what it was designed to do. Route (2)
(post-hoc q-argmax) remains available as a cheap bot-side guard on top of
this; route (3) (on-demand subgame resolving) is still deferred past v0.

## Q4 — How good can a *pure* (non-mixing) strategy be? (2026-07-12, open)

Forward-looking; a first cut is possible on the current subgames, the
whole-game number needs a full-game solve. Equilibrium truco mixes — the
bluff-balancing that makes a range unexploitable requires randomizing. A pure
(deterministic) strategy cannot balance, so it is exploitable in principle.
The question is *how much*: what is the lowest exploitability achievable by any
pure strategy, and how far above the mixed equilibrium's (~0) does it sit?

Why it matters. Humans cannot randomize reliably, so if a well-chosen pure
strategy is only slightly more exploitable than the mixed optimum, playing
deterministically costs a human almost nothing — and it is far easier to learn
and execute. The mining pass already found that ~65% of well-reached 11×11 card
decisions are *already* pure and only ~25% are genuine mixing (insight #7,
FINDINGS 2026-07-12), so the mixing is concentrated; Q4 asks what the best pure
strategy actually gives up by dropping it entirely. It also quantifies "how
much does mixing buy you in this game," which is interesting in its own right.

To settle it: (a) cheap upper bound available now — purify the current solve
(q-argmax everywhere) and measure the purified strategy's exploitability with
the per-infoset best-response machinery from plan 75; that is *a* pure strategy,
so it upper-bounds the *best* pure strategy's exploitability. (b) The true
minimum-exploitability pure strategy is a harder combinatorial search (choosing
one action per infoset to minimize worst-case exploitation) and is the real Q4;
on a full-game solve it gives the headline "cost of determinism" number.

## Q5 — A human-memorizable strategy: distill to a compact decision tree (2026-07-12, open)

Forward-looking; gated on a good full-game strategy to distill *from* (a full
solve, or a high-quality neural player — see plan 71, neural distillation). The
goal here is deliberately *not* to match equilibrium. It is to find a strategy
**simple enough that a human could memorize it** that still plays *decently
against humans* — explicitly not against a perfectly-adapting best-responder.
Hypothesis (owner): matching a bot's exploitation this way is likely
impossible, but "good enough versus humans" is plausible.

Why it matters / why it might work. Much of optimal play is already simple: the
mão-de-onze accept is a pure threshold (insight #2), ~65% of 11×11 card
decisions are pure (#7), and the mining surfaces human-legible rules of thumb
(lead low in ferro, always hoard the lone manilha, aggression escalates with
information). So a compact rule set may recover most of the value. Decision
trees / rule lists are one interpretable model class to fit against the target
strategy, sweeping the size↔fidelity tradeoff to find the smallest tree that
stays within an exploitability (or performance) bound.

The metric is itself a design question, and it is the crux of "vs humans, not
vs bots": exploitability (worst case against a best-responder) is the *wrong*
objective for this goal — a memorizable strategy will always be exploitable in
the worst case. The right objective is expected performance against a realistic
*human* opponent model (a population of human-like players), which the
distillation should optimize for and be evaluated against. This is the
maximally-compressed, human-facing end of the same spectrum as plan 71's
full-fidelity neural distillation.

To settle it: fit interpretable models (decision trees, rule lists, threshold
rules) to a full-game equilibrium or NN strategy; sweep model size vs fidelity;
evaluate the distilled strategy both by best-response exploitability (for the
theoretical ceiling) and, more importantly, against a human-like opponent model
(for the actual goal). A neat, self-contained exercise once a full-game target
exists.

**First target (owner, 2026-07-12): 11×11.** Distilling a memorizable strategy
for the mão-de-ferro deciding hand alone would already be a real win, and it is
the natural pilot: it is fully solved, self-contained (no raise ladder, no
onze, sudden death), and a compact "how to play the final hand" ruleset is
high-value on its own. It also sidesteps the hardest-to-compress machinery
(the raise ladder), so a small tree has the best chance of recovering most of
the value there.

## Q6 — When does the turn-up (vira) actually change optimal play? (2026-07-17, open)

The 2026-07-17 `compare-policies` measurements say the turn-up class barely
moves on-path play (reach-weighted mean TV 0.022-0.024, best-action agreement
~97%, at both 11x10 and 10x10), while the score moves it a lot (0.129, 86%).
But "barely" is an average over ~4-6% of play mass that DOES change — and
those rows are where the interesting card-blocking logic must live.

Owner intuition worth testing as a concrete pattern family: visibility of the
blocked rank should modulate aggression — e.g. "I hold a 2 and the turn-up is
a 3: the rank directly above my 2 is blocked, so fewer hands beat mine and the
opponent's raising range should thin." The tooling exists to answer this
cheaply: take the matched cross-TC row pairs whose TV is large (the divergent
tail), group them by (own hand class, public plays, history), and look for
systematic direction — does the policy raise/accept more when the turn-up
blocks ranks adjacent to its own holdings? A `compare-policies` extension that
dumps the top-K divergent rows with their info-set metadata (instead of only
aggregate stats) is the natural first instrument; the guide's Part II study of
optimal play is the consumer.

2026-07-18 update: the instrument exists — `solve compare-policies
--dump-divergent K [--dump-min-tv TV]` prints DIVERGENT_ROW lines (info-set
hand/history as action codes, both mixes), ranked by reach·TV when
--reach-weighted. Still open: run it on the 11x10 tc0-vs-tc1 pair (the
~2.2 GB converged artifacts don't fit local disk — temp-VM job) and read the
divergent tail for the blocked-rank-visibility pattern; then write the guide's
Part II vira chapter from what it says.
