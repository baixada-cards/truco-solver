# Plan 86 — Compute procurement for the 1v1 exact campaign

Status: DECIDED 2026-07-27 (procurement strategy locked; execution awaits
funding). Companion to plan 84 (whose round-cost pin produced every workload
constant used here) and plan 83 (the neural track that may reuse the same
capacity).

## Decision

**Rent, don't buy — Netcup root servers, ~$400 planning figure, $500
ceiling, for the full calibrated 1v1 exact grid at ε=0.01.** Buying hardware
is not competitive for this campaign alone and becomes plausible only if a
future 2v2 program projects ≥ ~12,000 additional effective CPU-hours (see
the 2v2 gate below).

An external procurement study (assessed and arithmetic-verified 2026-07-27;
summary below) compared hyperscaler spot, budget dedicated hosts,
marketplace CPU rental, and buy-use-resell. All of its internal math checks
against the plan-84 measured constants.

## Workload constants (measured — plan 84 / SOLVER_BENCHMARKS 2026-07-27)

- Campaign: **~3,000–4,200 effective 16-thread box-hours** warm-started
  (central 3,600); ~6,700 if warm starts underdeliver.
- Peak job RSS **85.6 GiB** → 96–128 GB boxes (96 GB tier usable only if a
  live test shows the guest RAM is really GiB-class; decimal-GB guests leave
  ~3.8 GiB headroom — test before relying on it).
- ~250 GB NVMe per job (arena packs ~80 GB + 15.7 GiB checkpoints + scratch).
- Interruption cost at the deepest cells ≈ 1.9 h re-init + one ~10 min round
  → big cells prefer non-preemptible capacity.
- GCP spot reference (measured): n2d-highmem-16 at ~$0.24/hr →
  **$720–1,600 all-in**.

## The selected plan (risk-controlled central case, ~$365–378)

1. **1× Netcup RS 16000 G12** (EPYC 9645 Turin, 24 dedicated cores, 128 GB
   ECC, 4 TB NVMe, ≈$138/mo ex-VAT) for one month — runs ALL high-memory job
   classes first.
2. **1× Netcup RS 12000 G12** (20 cores, 96 GB, 3 TB NVMe, ≈$98/mo ex-VAT)
   for two months — runs everything else, contingent on the 96 GB
   guest-memory test passing.
3. Azure Spot only for a fractional final tail ($0–8).
4. ~2 TB result storage for ~1 month: $30–35 (zero-egress class).

Variants: all-RS 16000 ≈ $445–458 (safe, still ~half of GCP); RS 12000-only
base ≈ $325–416 if the 96 GB test passes. Prudent authorization: **$500**.

## Verify at purchase time (not yet independently confirmed)

- [ ] Live Netcup prices/stock/setup fees and month-to-month terms
      (study cited netcup.com/en/server/root-server, checked 2026-07-27).
- [ ] VAT status for a non-EU (Brazil) customer — ex-VAT prices assumed.
- [ ] **The load-bearing assumption: 1.7–1.9× effective throughput vs the
      GCP n2d baseline** (16 dedicated Turin cores vs 16 SMT vCPUs).
      First hour on any rented box: re-run the plan-84 pin benchmark
      (resume the retained 0×0 checkpoint, 3 rounds) and replace the
      multiplier with a measurement before committing the fleet plan.
- [ ] 96 GB tier: run the biggest job class once with an NVMe swapfile
      armed; sustained swapping disqualifies the tier.
- [ ] Netcup KVM disk/IO behavior under the arena-cache mmap load.

## Why buying loses for 1v1 alone (and the 2v2 gate)

Used-server cash cost over the campaign ≈ $700–800 ($1,150–1,300 fully
loaded with labor/resale/failure risk) vs $365–450 rental. Break-even on the
buy premium: ~6,500–8,000 extra effective hours cash-basis (verified),
~12,000 as a practical risk-loaded threshold. Decision table:

| expected additional 2v2 CPU work | decision |
|---:|---|
| < 3,000 h | rent |
| 3,000–8,000 h | probably rent |
| 8,000–15,000 h | buying plausible |
| 15,000–25,000 h | buying probably justified |
| > 25,000 h | ownership strongly favored (if RAM fits) |

If buying ever triggers: single-socket EPYC (Rome/Milan), 24–48 cores,
**384–512 GB ECC** (three-four concurrent 96–128 GB jobs — a 128 GB machine
runs only ONE), 2–4 TB NVMe. Never buy before the 2v2 feasibility benchmark
fixes the natural 2v2 worker's RAM class (128 vs 256 vs 512 GB).

## 2v2 feasibility benchmark (prerequisite for any purchase)

Bounded phase, run on rented capacity: specify the 4-seat information model
(seating/turn order, partner visibility, raising authority, signaling
rules); build the smallest correct four-hand engine; census one turn-up
class at one shallow score (deals, abstract deals, nodes, info sets per
seat/team, round-1 public boundaries, bytes/node); measure a sampled solve
(peak RSS, build time, nodes/s, checkpoint size, subgame distribution); then
project the mão-de-onze-row + 10×10-analog teacher cost.

Theory caveat recorded: 2v2 is a two-TEAM game with per-seat private
information. Modeling a team as one player would illegally share hands;
modeling four players with shared team utilities loses the two-player
zero-sum CFR guarantees. The equilibrium/certification definition (TMECor-
style or otherwise) must be chosen deliberately before any solver work. What
transfers from 1v1: trunk decomposition, safe re-solving, independent
workers, f32 accumulators, checkpointing, error budgets, exact-shallow-tiers-
as-teachers. What does not: the engine's hardcoded two-player structure
(reach arrays of length 2, `1 - player` opponents, p0-perspective utilities,
two-BR exploitability) and any trained 1v1 policy (pretraining value only).

## Execution order when funded

1. Rent RS 16000 → hour 1: throughput pin (replace 1.7–1.9× with a number).
2. 96 GB test on RS 12000 → pick the cheap or safe fleet shape.
3. Solve in dependency order (shallow → deep; each band produces real match
   values and warm starts for the next — kills the synthetic-mv caveat).
4. ε=0.01 stopping rule everywhere; checkpoints retained; teacher-grade
   2.5e-4 stays a later resume-extend purchase (~10× iterations, additive).
