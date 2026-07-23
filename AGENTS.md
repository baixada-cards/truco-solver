# Agent Instructions

## Repository Purpose

This repository owns Baixada's Truco CFR solver, policy-format producer,
autoresearch harness, and living research record. It consumes the authoritative
engine from `baixada-cards/truco-engine`; it does not redefine game rules.

## Canonical Boundaries

- `crates/truco-solver` owns training, tree construction, certification,
  checkpoints, exports, and solver CLIs.
- `crates/truco-policy-format` owns the stable runtime interchange contract:
  abstraction/action codes, deterministic information-set keys, TPB1 files,
  manifests, and compatibility tests.
- Runtime bots may consume `truco-policy-format`; they must not depend on the
  full solver.
- `engine.lock.json` and the root Cargo dependency pin the same immutable
  `truco-engine` revision. Never replace it with a moving branch.
- Hosted services, provider bots, frontend code, deploy credentials, personal
  cloud inventories, and trained artifacts do not belong here.

## Workflow

- Run `make check` before wrapping up a change.
- Use `sfw` for public-registry dependency fetches.
- Keep `Cargo.lock` and `autoresearch/uv.lock` current and use locked/frozen
  installs in CI.
- Sign commits.
- Use a dedicated clone or worktree for autoresearch: its runner intentionally
  creates and may discard commits.
- Never commit checkpoints, treepacks, teacher exports, TPB1 policy bundles,
  secrets, `.env` files, or cloud inventories.

## Compatibility

- Existing TPB1 v1 bytes and deterministic key vectors remain readable.
- A key, action-code, record-layout, ordering, or probability-encoding change
  requires a new policy-format version and golden compatibility fixtures.
- Solver compatibility re-exports exist for source continuity; new runtime
  code should import `truco-policy-format` directly.

## Verification

- Rust: formatting, Clippy with warnings denied, and all workspace targets.
- Policy format: manifest validation plus golden key and byte vectors.
- Autoresearch: Ruff and its Python test suite.
- Dependency boundary: `cargo tree -p truco-policy-format` must contain no
  `truco-solver`, and runtime consumers must not introduce the inverse edge.

## Living Documentation

Keep these files current whenever their subject changes:

- `SOLVER_PLAN.md` — architecture, status, and planned work.
- `SOLVER_BENCHMARKS.md` — append-only benchmark, convergence, cost, and
  hardware results.
- `RESEARCH_NARRATIVE.md` — decisions, attempts, findings, and open questions.
- `EXACT_SOLVING.md` — exact-solve feasibility conclusions.
- `autoresearch/README.md` and `autoresearch/program.md` — harness architecture
  and agent interface.

Do not leave meaningful benchmark results only in logs.
