# 84 — Composed cheap-solve benchmark infra

## Goal

Build and benchmark the full composed cost stack for exact solving — every
accepted or promising lever running **together**, not measured in isolation:

- ε=0.01 relaxation (accepted)
- same-band warm starts (accepted)
- asymmetric per-player raise pruning (accepted, `TreeRules::AsymmetricRaisePrune`)
- cross-TC transfer-and-certify (accepted as portfolio)
- **f32 accumulators** (plan 79 Phase 6, never A/B'd — this plan implements it)
- **CFR-D / safe subgame re-solving** (EXACT_SOLVING.md §6 thread 6 — prototype here)
- subgame-parallel distribution on commodity spot (replaces deal-sharding's
  all-reduce with embarrassingly-parallel subgame jobs)
- mmap treepack streaming (already built; carried as *insurance only* — if
  subgames are small, nothing needs paging)

Context: the 2026-07-17 lossless-prune audit (SOLVER_BENCHMARKS.md) closed the
pruning family — >99% of the expensive symmetric deep cells is live
betting/bluffing tree, immune to dominance/isomorphism/saturation. Remaining
savings live in representation + decomposition. Target trajectory:
$12–16K → ~$3–4K (central) → ~$1–1.5K (floor) for the full grid at ε=0.01.

## Why this matters beyond 2-player

If the composed stack works, the same pattern is the feasibility path for
**4-player truco**: the tree is far bigger, but the decomposition pattern
(trunk + safe subgame re-solving + tiny independent spot jobs), f32
representation, and the ε-budget arithmetic all transfer. Concretely, the
first 4p milestones would be the **mão-de-onze row + 10×10-analog cells** —
exactly the cheap tiers 2p solved first — which could seed a supervised /
neural full-game policy (plan 83's approach, applied to 4p). The engine and
abstraction do NOT transfer (4p needs partnership info sets, signaling, new
deal enumeration); do not widen this plan's scope into engine work.

## Phases

### Phase 1 — f32 accumulators (`accum-f32` cargo feature)

`DenseAccum { regret, strategy, pending, last }: Vec<f64>` → `Vec<Acc>` with
`type Acc = f32` under the feature, `f64` default. Rules:

- **Checkpoint/strategy format stays f64** — widen on write, narrow on load.
  Artifacts from both builds remain byte-compatible and comparable.
- `LocalAccum` (thread-local parallel buffer) stays f64: transient,
  and keeping the per-sweep sums f64 limits the rounding to one
  narrow-per-fold, mirroring how `prune_regret: Vec<f32>` already works.
- All traversal math (reach probs, node values, regret deltas) stays f64;
  narrowing happens only at accumulator read/write boundaries.
- Accuracy contract is plan 79 Phase 6 verbatim: identical tree counts,
  exact-BR ε + game value vs f64 control at checkpoints, resume round-trip,
  ≤1e-5 abs ε drift on cheap deterministic cases, ≤1% relative at production
  target, measured RSS win.

Expected win: ~½ of accumulator RAM (the ~⅓ pool), ~15–17% of a solve job's
total; the real value is setting the RAM floor for decomposed subgame jobs
and halving CFR-D boundary-value tables.

### Phase 2 — per-infoset BR-value export — ALREADY BUILT ✅

Discovered during implementation (2026-07-17): the export already exists.
`best_response_gaps_for_profile` (cfr.rs) exposes the bottom-up pass's
per-infoset values as `InfoSetBestResponse { table_idx, br_value, weight }`,
and `compact-br --br-gaps --br-gap-out PATH` writes
`(table_idx, br_value, eq_value, gap, weight)` rows via
`teacher_export::save_br_gaps` (the 2026-07-14 pilot artifact,
SOLVER_BENCHMARKS.md). This IS the CFR-D gadget's boundary input and the
per-infoset exploitability metric. Phase 3 consumes it as-is; only verify the
`weight` field is the counterfactual reach the gadget construction expects.

### Phase 3 — CFR-D prototype at ground truth (10×10)

The validation asset: exact 10×10 checkpoints (`10x10-full-20260709/d0/`,
ε≈2.5e-4) provide ground truth no published CFR-D implementation had.

- Boundary: public state after the round-1 betting/play sequence (public
  raise history + face-up plays partition deals into range-consistent
  subgames).
- Gadget: Burch–Johanson–Bowling opt-out construction from Phase 2 boundary
  values.
- Harness: solve trunk → re-solve each subgame behind the gadget →
  compose → `compact-br` the composed strategy on the full tree.
- Acceptance: composed exploitability ≤ blueprint ε + designed re-solve
  budget; compare against the known exact solve. Measure peak RSS of the
  largest single subgame (the number that decides commodity-box feasibility).

### Phase 4 — trunk-CFR loop: CFVs without a full solve (CFR-D proper)

(Renumbered 2026-07-18: this was resolve.rs's "future work (Phase 4)"; the
fleet-shape items previously here fold into Phase 5, since Phase 5 is where
they're first exercised.)

Phase 3 validated the gadget in ORACLE MODE — boundary CBVs extracted from
an existing full solve. Phase 4 removes the oracle: solve the trunk and the
subgames in alternation, from scratch, no blueprint anywhere.

Design (Burch–Johanson–Bowling decomposition, pragmatic schedule):

- **Trunk** = everything above the round-2 boundary (reuse
  `subgame::collect_boundary` unchanged). Trunk sweeps treat boundary nodes
  as terminals whose values come from a **cached per-boundary-node value
  pair** `(v0, v1)` — the subgame's value under both players' CURRENT
  subgame average profiles — refreshed by one eval sweep after each
  subgame re-solve round (trunk reads are then O(1) per crossing).
- **Subgame re-solves during the loop run WITHOUT the gadget** (per BJB:
  the gadget belongs to final recovery, not the trunk phase): persistent
  per-subgame accumulators (warm across rounds), root distribution
  `π_c · π^trunk-current` refreshed each round.
- **CBV accumulation:** running reach-weighted average of each opponent
  view's counterfactual value under the subgame's current profile, per
  round — this becomes the Terminate value for final recovery.
- **Final recovery:** gadget re-solve per subgame (Phase 3 machinery
  as-is) from the ACCUMULATED average CBVs + trunk-average root weights;
  compose trunk average + re-solved subgames; certify with exact BR.
- **Schedule knobs:** `--rounds N --trunk-sweeps T --subgame-iters R`
  (+ final re-solve iters). Defaults to probe: (30, 3, 3) and pure
  alternation (90, 1, 1).
- **The measurement:** total node-visits / (monolithic 90-iter visits) —
  the re-solve multiplier that decides Phase 5's cost model — plus the
  usual certificate and game-value-vs-ground-truth comparison.

Acceptance: from-scratch composed 10×10 tc0/d0 certifies at ε ≤ 0.01
(production target; the teacher blueprint's 2.5e-4 is NOT the bar — 120-iter
re-solves already showed composed ≈ 1.7e-3), game value matches the known
solve to ~1e-3, multiplier measured. Prototype memory model (full-size
accumulators, full trees in RAM) stays — 75 GB boxes fit 10×10; trunk-only
arenas are Phase 5 work.

### Phase 5 — the composed benchmark (+ fleet shape)

One symmetric deep cell — the cells nothing else could touch — end-to-end on
commodity spot only (≤64 GB boxes), all levers on. Absorbs the fleet-shape
items (subgames given trunk CFVs are independent: job-in/CFV-out on the
existing SPOT+watcher pattern; warm re-solves; per-subgame cross-TC
transfer-and-certify) and needs trunk-only arena builds + per-subgame
arena builds to hit the ≤64 GB target:

- Candidate: 1×1 tc0 (or 0×0 if Phase 3 RSS numbers say it fits).
- ε-budget designed up front: trunk ε + Σ subgame re-solve ε ≤ 0.01 with
  explicit allocation, certified by compact exact-BR at the end.
- Record: $/cell composed vs the $53 premium-box estimate, peak RSS per job,
  wall, iteration counts (also converts the §4 deep-tier iteration-count risk
  from assumption to measurement).
- Fermi gate: composed deep cell ≤ ~$10 ⇒ full grid ~$3–4K is real.

## Non-goals

- No mmap work (built; insurance only).
- No lossy abstraction (plan 83 territory).
- No 4p engine work (motivation only, recorded above).
- No fleet credit claims until a Phase 5 cell certifies (SOLVER_BENCHMARKS
  discipline: measured, dated entries).

## Status

- [x] Phase 1 — f32 accumulators (`accum-f32`) — implemented 2026-07-17;
      local A/B gate + cross-build checkpoint compatibility PASS
      (`SOLVER_BENCHMARKS.md` 2026-07-17). Production-scale ε drift check
      rides along with Phase 5.
- [x] Phase 2 — BR-value export — discovered already built
      (`--br-gaps --br-gap-out`, 2026-07-14 pilot).
- [x] Phase 3 (core) — CFR-D built and verified locally 2026-07-18:
      `subgame.rs` (round-2 boundary via build-replay walker, public-state
      grouping, reach/CBV extraction), `cfr.rs::resolve_subgame` (BJB
      Terminate/Follow gadget over existing packed subtrees, SyncCFR+
      freeze-then-fold discipline), `resolve.rs` (orchestration, composition,
      certification, repair harness). Tests: boundary partition/span/view
      invariants; composed ε ≤ blueprint ε + slack (measured: composed
      0.005745 BEAT blueprint 0.007446); and the core CFR-D claim — a
      corrupted subgame (ε 0.113) repaired to 0.007449 from the boundary
      summary alone, 99.997% recovery (`SOLVER_BENCHMARKS.md` 2026-07-18).
      103/103 crate tests green. **Phase-3 tail COMPLETE (2026-07-18):**
      `solve resolve-subgames` CLI + the production 10×10 ground-truth run
      (all 140k deals, real fleet blueprint, `--legacy-tree`). Repair from
      boundary summary alone: 0.008782→0.000248437 vs blueprint
      0.000248280 (+1.6e-7, 99.998% recovery). 495 subgames; largest
      re-solve unit 0.52% of the cell. Total spend $3.45 of a $5 cap; two
      production bugs (O(subgames×BR) CBVs; legacy-tree action mismatch)
      found by backstops/diagnostics and fixed. `SOLVER_BENCHMARKS.md`
      2026-07-18 has the full table.
- [x] Phase 4 — trunk-CFR loop (CFVs without a full solve) — COMPLETE
      2026-07-21. From-scratch 10×10 tc0/d0: raw certificate **0.001122**
      (9× under the 0.01 gate), gv Δ 5.5e-5 vs teacher ground truth.
      Production plateau (~0.018 across r90/r150/r300) diagnosed by the
      three-certificate run and fixed: intra-round couplings must read the
      CURRENT regret-matching iterate, not the lagging average
      (`DenseAccum::current_row`; commits 86984b3, 53823c9). Toy probe
      after fix beats the monolithic control (0.000239 vs 0.000354).
      4 runs, $3.35. `SOLVER_BENCHMARKS.md` 2026-07-21.
- [x] Phase 5 (probe) — COMPLETE 2026-07-23, $7.55 of a $10 cap. 0×0
      solves + certifies on ONE 128 GB commodity box (n2d-highmem-16 spot,
      $0.24/hr): init 88 min, parallel certificate 10.6 min, raw ε 0.246
      @ 3 rounds (on the 1/t curve), peak RSS 124.5 GiB, 15.74 GiB
      checkpoint resumed across zones/SKUs/provisioning models. Three
      production bugs found+fixed (boundary-state OOM → replay seeds
      ce4f05d; sequential cert → subgame-parallel bcb24234; post-cert
      artifact OOM → open, fix pattern known). ε=0.01 extrapolation:
      $9–45/cell·tc·dealer at current code, $2–5 with arena caching +
      R>1 (unbuilt). `SOLVER_BENCHMARKS.md` 2026-07-23.
- [x] Phase 5 (tail, ENGINEERING) — COMPLETE 2026-07-27, $0, local
      (`SOLVER_BENCHMARKS.md` 2026-07-27):
      - [x] Streamed composed artifact — fixes the post-certificate OOM.
            `deep_solve` takes an artifact sink and streams rows subgame by
            subgame; no whole-profile map, no cloned rows, no full-arena
            rebuild. Measured 4.35× lower peak RSS at 8×8/2,000 deals, output
            content-identical (max TV 0.000000 over 3.87 M rows).
      - [x] Arena disk cache (`--arena-cache DIR`, default ON at
            `<checkpoint>.arenas`) — 3.3–3.7× wall per round at jobs=1 for
            0.4–7% peak RSS, against 2.0–2.3× RSS for `--keep-arenas`.
            Bit-identical certificates.
      - [x] Resume-EXTEND — `--rounds` may grow on resume; extending a
            checkpoint is bit-identical to having asked for the longer run up
            front. This is the mechanism the "ε=0.01 now, 2.5e-4 later"
            strategy assumed but did not have.
      - [x] `--cert-jobs` (default `min(jobs, 8)`) — bounds the memory-critical
            certificate pool separately from the round pool.
      - [x] Key-map compaction: `SubgameState::key_to_local` (~757 M entries
            ≈ 30 GB at 0×0) DELETED rather than compacted — streaming
            composition was its only consumer, and the sweeps key off the
            local `table_idx`, not the map.
- [ ] Phase 5 (tail, PRODUCTION SCALE) — **awaiting funding**; nothing below
      is blocked by code any more:
      - [ ] The $0.50 spot round-cost pin at 0×0 (3 rounds, resume from the
            retained 15.74 GiB checkpoint) — turns the 15–75 min/round band
            into a measurement, now also measuring the arena cache's effect
            at real scale rather than extrapolating the local 3.3×.
      - [ ] The r150 certified 0×0 cell (ε≈0.01), and the r1500 extension to
            ε≈2.5e-4 on top of it — the extension is now genuinely
            incremental (resume-extend), so the two can be bought separately.
      - [ ] R>1 amortization sweep (`--subgame-iters` > 1) — untouched by this
            tail; still an open cost lever.
