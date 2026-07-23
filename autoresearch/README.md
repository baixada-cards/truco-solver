# Truco CFR Solver Autoresearch

Automated algorithm research for the Truco CFR solver, inspired by
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## Concept

An LLM autonomously iterates on the CFR algorithm, modifying a single Rust
file (`cfr_experiment.rs`), running a fixed benchmark, and comparing
exploitability. Better results are kept in the experiment history; worse
results are discarded after the run, with logs and source snapshots preserved.

This package is specifically for **CFR autoresearch**. Future autoresearch loops
elsewhere in the repo should get their own names and harnesses instead of
inheriting this solver-specific framing. The checked-in harness is the public,
portable control surface. Personal VM inventory and cloud launch automation
live in the private `baixada-ops` repository.

## Architecture

```
autoresearch/
├── README.md              ← you are here
├── program.md             ← instructions for the LLM agent
├── harness.py             ← orchestrator: compile, run, measure, log, live heartbeats
├── llm.py                 ← LLM client (Anthropic or OpenAI API)
├── runner.py              ← main loop: propose → run → evaluate → keep/discard
├── results.tsv            ← experiment log (tracked and preserved across resets)
└── pyproject.toml         ← Python dependencies
```

**In the solver crate:**

```
crates/truco-solver/src/
├── cfr_experiment.rs      ← THE FILE the LLM modifies (algorithm core)
├── cfr.rs                 ← tree building, exploitability, and solve helpers
├── ...                    ← everything else (immutable during experiments)
└── bin/
    └── experiment.rs      ← binary entry point for 10-min benchmark runs
```

## Key design decisions

1. **Single mutable file.** The LLM only edits `cfr_experiment.rs`. This file
   contains the iteration logic: how regrets are updated, how strategies are
   computed, any discounting or weighting. Everything else (tree building,
   deal enumeration, exploitability computation, info set management) is
   immutable infrastructure.

2. **Current fixed time budget.** In the direct-provider harness, every
   experiment runs for exactly 10 minutes of CFR iterations (wall clock,
   excluding tree build and exploitability). This makes experiments comparable
   regardless of algorithmic changes. The planned Claude Code runner should add
   max-iteration and dollar-budget limits around the same benchmark discipline.

3. **Single metric.** Exploitability after 10 minutes. Lower is better.

4. **Git-based versioning.** Each experiment is committed, logged, and then
   either kept or reverted based on the measured result.

5. **Planned Claude Code backend.** The current runner calls Anthropic/OpenAI
   APIs directly and asks for a complete replacement `cfr_experiment.rs`. The
   backlog now calls for a shipped coding-agent backend that runs one iteration
   through Claude Code first, because `claude -p --max-budget-usd <amount>`
   provides a native per-iteration dollar cap.

6. **Runner-owned control loop.** The coding agent should propose and edit the
   CFR experiment surface, but the runner should own max iterations, overall LLM
   budget, per-iteration budget, source replacement, acceptance/rejection, result
   logging, git commits for accepted candidates, and VM lifecycle. MLflow should
   record every attempted run, but git remains the accepted-candidate lineage and
   human-review layer.

7. **Persistent private MLflow tracking.** The planned runner should log
   parameters, metrics, candidate source, diffs, prompts, Claude outputs,
   benchmark logs, and final artifacts to a persistent personal-server MLflow
   deployment reachable only over Tailscale. VM runs should use
   a project-level tracking URI that the runner passes through to MLflow as
   `MLFLOW_TRACKING_URI`, so the tracking database and artifacts do not need to
   be moved over SSH after each run. A file-backed MLflow store remains useful
   as a local/offline fallback while the personal server is being set up.

   **Implemented now (direct harness):** `runner.py` logs every candidate
   (`baseline`/`keep`/`discard`/`crash`) to MLflow via `mlflow_tracking.py` —
   params (provider, model, time budget), metrics (exploitability, iterations,
   wall seconds, accepted), a `status`/`campaign_id` tag set, and the per-commit
   `.rs` + `.log` artifacts. It defaults to a durable local file store
   (`autoresearch/mlruns`) and switches to a server when `MLFLOW_TRACKING_URI`
   (+ optional `MLFLOW_TRACKING_USERNAME`/`PASSWORD`) is set. Resolve any
   `op://` references at the process boundary with `op run`; never persist
   resolved credentials in this repository. Tracking is **fail-safe**: if
   MLflow is missing or unreachable, the research loop logs a warning and
   continues. Disable per-run with `runner.py --no-mlflow`.

8. **Safe environment handling.** Non-secret defaults live in committed example
   env files such as `autoresearch/.env.example`. Local `.env` files stay
   ignored. If a local env value is an `op://...` 1Password reference, the ops
   runner should resolve it in memory and inject the resolved value directly into
   the child process environment without writing it to disk, prompts, logs, or
   MLflow params.

9. **Operations boundary.** This public repository owns the experiment harness.
   Private cloud launch, monitoring, and artifact retrieval consume this
   repository from `baixada-ops`; there is intentionally no public web control
   plane or checked-in live inventory.

## Planned launch controls

The CFR autoresearch launcher should grow from a provider/model prompt into a
small configuration panel:

- max iterations
- max iterations counts attempted proposals, not accepted improvements
- max overall LLM budget in US dollars, counting LLM spend only
- max per-iteration LLM budget in US dollars, passed to Claude Code with
  `--max-budget-usd`
- Claude worker model, defaulting to `sonnet`
- Claude worker effort, defaulting to `high`
- advisor disabled in the first shipped workflow
- Claude model choices, starting with the small hardcoded allowlist `sonnet` and
  `opus`
- Claude effort choices, using the CLI-supported values `low`, `medium`, `high`,
  `xhigh`, and `max`
- exploration mode:
  - **backlog-guided:** encourage variations of the solver ideas already listed
    in the backlog, including pruning, lazy CFR, DCFR+, PDCFR+, sampling, and
    parallelism experiments
  - **free exploration:** let Claude propose novel CFR changes within the same
    mutable-file and benchmark constraints

The chosen launch configuration should be copied to the VM, recorded in the run
log, and appended to experiment results so later comparisons include the budget,
model, effort, and exploration mode that produced them.

Advisor is intentionally out of the first shipped workflow. Interactive Claude
Code has `/advisor`, and local version `2.1.128` contains a hidden
`--advisor <model>` option, but `claude -p --advisor opus ...` currently rejects
the flag. A separate `opus`/`xhigh` advisor call is scriptable, but it adds budget
and orchestration complexity before we know it improves CFR outcomes.

Claude Code permissions should lock the proposal step down to the experiment
file. The runner should allow reads and only allow `Edit`/`Write` for
`crates/truco-solver/src/cfr_experiment.rs`, deny Bash and `.env*` reads, then
reject any attempt whose post-run diff touches a different file.

The private operations layer may expose an MLflow dashboard. When using the
local file fallback, bind the UI to loopback only; public cloud ports and live
tracking credentials are outside this repository.

## Current status

The recorded autoresearch results live in `autoresearch/results.tsv`.

Best kept runs so far:

| Commit | Exploitability | Iterations | Iter wall time | Description |
|--------|---------------|------------|----------------|-------------|
| `4b1300b` | 0.663562 | 25 | 861.7s | baseline |
| `38fd48c` | 0.157519 | 70 | 866.9s | DCFR+ hybrid with regret clamping and discounting |

These numbers are for the autoresearch harness's fixed 10-minute iteration
budget. They are useful for comparing autoresearch proposals to each other, but
they are not directly comparable to the longer manual benchmark runs described
in `SOLVER_BENCHMARKS.md`.

## The mutable file interface

`cfr_experiment.rs` must implement:

```rust
/// Called once before iterations begin. Initialize any per-solve state.
pub fn init(num_info_sets: usize) -> ExperimentState;

/// Called each iteration. Traverse all trees and update regrets/strategy.
/// Returns the iteration number (for logging).
pub fn iterate(
    state: &mut ExperimentState,
    iteration: u64,
    traversing: Player,
    prebuilt: &PrebuiltTrees,
    table: &mut StrategyTable,
    score: &Score,
    match_values: &MatchValueTable,
);

/// Called after each iteration. Apply any post-iteration processing
/// (discounting, pruning, etc.)
pub fn post_iterate(
    state: &mut ExperimentState,
    iteration: u64,
    table: &mut StrategyTable,
);
```

The harness calls these in a loop until the 10-minute budget expires, then
computes exploitability and reports the result.

Long-running phases stream live feedback. Compile, LLM proposal, and solver
execution all emit periodic heartbeat lines so tmux sessions and `runner.log`
keep showing visible progress instead of going silent for minutes.

Each committed experiment also saves a source snapshot in
`autoresearch/logs/<commit>.rs`, so discarded or crashed runs are still
recoverable even if they are reverted from git history later.

## What the LLM can change

Everything inside `cfr_experiment.rs`:
- Regret update formula (CFR+, DCFR, linear CFR, novel ideas)
- Strategy computation from regrets
- Discounting parameters and schedules
- Post-iteration processing (pruning, rescaling)
- Per-info-set auxiliary state
- Warm-starting heuristics
- Any novel algorithmic ideas

## What the LLM cannot change

- Tree building (`game_tree.rs`)
- Deal enumeration (`abstraction.rs`)
- Exploitability computation (`cfr.rs::compute_exploitability`)
- Info set representation (`info_set.rs`)
- Strategy table structure (`strategy.rs`)
- The harness itself (`harness.py`, `runner.py`)
- The evaluation metric (exploitability)

## Running locally

```bash
# In a gitignored autoresearch/.env:
ANTHROPIC_API_KEY=op://<vault>/<item>/credential

# Optional explicit model override
export AUTORESEARCH_MODEL="claude-opus-4-6"

# Run the research loop
cd autoresearch
op run --env-file=.env -- uv run runner.py

# Or run a single experiment manually
cd /path/to/truco
cargo run --release -p truco-solver --bin experiment
```

Provider/model selection rules:

- If `ANTHROPIC_API_KEY` is set, the runner prefers Anthropic and queries
  `GET /v1/models` to choose the best available preferred model.
- If `OPENAI_API_KEY` is set and Anthropic is absent, the runner uses OpenAI.
- If `AUTORESEARCH_MODEL` is set, it overrides the provider default.
- Without an override, the OpenAI default is `o3-mini`.

## Cloud execution

Cloud workers clone a signed revision of this repository and run the same
commands. Provider-neutral worker contracts may be documented here, but
project IDs, instance names, deploy keys, service accounts, buckets, retention
state, and launch commands are private operations data owned by
`baixada-ops`.
