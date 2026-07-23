# Truco Solver

Research-grade counterfactual-regret-minimization tooling for Baixada's
two-player Brazilian Truco.

This repository has two deliberately separate Rust crates:

- `truco-solver` builds abstract game trees, trains and certifies CFR policies,
  manages checkpoints, exports Study data, and runs experiments.
- `truco-policy-format` is the lightweight, versioned interchange contract
  shared with runtime bots. It owns abstract action codes, deterministic
  information-set keys, the TPB1 policy file, and its bundle manifest schema;
  it contains no CFR implementation.

The authoritative rules engine lives in
[`baixada-cards/truco-engine`](https://github.com/baixada-cards/truco-engine).
This workspace pins one exact engine commit in both
[`Cargo.toml`](Cargo.toml) and [`engine.lock.json`](engine.lock.json). Moving
branch dependencies are not accepted.

## Layout

| Path | Responsibility |
| --- | --- |
| `crates/truco-solver` | CFR algorithms, tree building, exact best response, storage, exports, and CLIs |
| `crates/truco-policy-format` | Solver/runtime policy contract and compatibility tests |
| `autoresearch` | Bounded LLM-driven CFR experiment harness |
| `SOLVER_PLAN.md` | Architecture, implementation status, and open work |
| `SOLVER_BENCHMARKS.md` | Append-only benchmark and cost record |
| `RESEARCH_NARRATIVE.md` | Decisions, failed paths, results, and open questions |
| `EXACT_SOLVING.md` | Exact-solve feasibility and operational conclusions |
| `QUESTIONS.md` | Focused research questions |
| `plans` | Solver-specific implementation and research plans retained with their subject |

Hosted matches, provider integrations, product UI, deployment credentials, and
live infrastructure inventories intentionally live elsewhere.

## Development

Prerequisites:

- stable Rust with Clippy and rustfmt;
- Python 3.12+ and `uv` for autoresearch;
- [Socket Firewall Free](https://docs.socket.dev/docs/socket-firewall-free) for
  public-registry fetches.

Install locked dependencies:

```sh
make sync
```

Run the complete repository gate:

```sh
make check
```

Useful direct commands:

```sh
cargo test -p truco-policy-format --locked
cargo test -p truco-solver --lib --locked
cargo run -p truco-solver --bin solve -- compare --iters 100
cargo run -p truco-solver --bin viewer
```

## Policy contract

`truco-policy-format` is public API even while the surrounding solver remains
pre-1.0. Existing TPB1 v1 bytes and deterministic information-set keys must
remain readable. A change to key derivation, action codes, record layout,
ordering, or probability encoding requires a new format version and golden
compatibility fixtures.

Runtime bots should depend on this small crate at a signed tag or full commit;
they must not depend on `truco-solver`.

## Artifacts and operations

Trained policies can be gigabytes, and checkpoints can contain private or
expensive research state. They are never committed to this repository or
published implicitly through Actions. Deployment fetches immutable,
checksum-verified artifacts from private object storage.

The repository includes generic local research code. Personal cloud project
names, VM inventories, deploy keys, credentials, retention state, and live
launch configuration belong in the private `baixada-ops` repository.

## Autoresearch

The autoresearch loop intentionally creates and may discard Git commits. Run it
only in a dedicated clone or worktree and read
[`autoresearch/README.md`](autoresearch/README.md) first.

## Versioning

Repository releases follow Semantic Versioning. Policy-format compatibility is
called out separately in each release because its wire-format stability is
stronger than the solver CLI's pre-1.0 stability.

## License

MIT. See [`LICENSE`](LICENSE).
