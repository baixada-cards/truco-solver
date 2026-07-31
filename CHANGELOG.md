# Changelog

All notable changes to this repository are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Upgraded `rand` 0.8 → 0.9.5 (includes the 0.9.5 `UniformChar`
  deserialization memory-safety fix) and `bincode` 1.3 → 2.0.1.
- All bincode traffic now goes through the `bincode_v1` wrapper module, which
  pins bincode 2's `config::legacy()` so every artifact keeps bincode 1's
  exact wire format. Verified byte-identical round-trip against real
  pre-migration GCS artifacts (match-value table, streamed 10x10 checkpoint,
  deep-solve checkpoint header) plus golden byte vectors in-tree.

## [0.1.0] - 2026-07-23

### Added

- CFR solver, exact best-response certification, checkpoint/storage formats,
  Study export tooling, and experiment entrypoints.
- Solver-independent `truco-policy-format` crate with TPB1 v1, bundle manifest
  schema, safe parsing, and golden compatibility vectors.
- Living solver plan, benchmark ledger, research narrative, and exact-solving
  record.
- Generic CFR autoresearch harness with locked Python dependencies.
- Exact `truco-engine` revision pin and locked, SHA-pinned public CI.

[Unreleased]: https://github.com/baixada-cards/truco-solver/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/baixada-cards/truco-solver/releases/tag/v0.1.0
