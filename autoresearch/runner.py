#!/usr/bin/env python3
"""Main experiment loop: propose → run → evaluate → keep/discard.

Usage:
    cd autoresearch
    op run --env-file=.env -- uv run runner.py [--model MODEL] [--budget SECS]
    AUTORESEARCH_MODEL=claude-opus-4-1-20250805 uv run runner.py [--budget SECS]
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from harness import (
    PROJECT_ROOT,
    ExperimentResult,
    format_duration,
    read_experiment_file,
    run_experiment,
    start_heartbeat,
    stop_heartbeat,
    write_experiment_file,
)
from llm import LLMConfig, detect_config, propose_experiment
from mlflow_tracking import Tracker

RESULTS_FILE = Path(__file__).parent / "results.tsv"
PROGRAM_MD = Path(__file__).parent / "program.md"
LOGS_DIR = Path(__file__).parent / "logs"

TSV_HEADER = "commit\texploitability\titerations\twall_secs\tstatus\tdescription\n"


def log(message: str = "") -> None:
    """Print a line immediately, even when stdout is piped through tee."""
    print(message, flush=True)


def init_results():
    """Create results.tsv with header if it doesn't exist."""
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(TSV_HEADER)


def append_result(
    commit: str,
    result: ExperimentResult,
    status: str,
    description: str,
):
    """Append one row to results.tsv."""
    expl_str = (
        f"{result.exploitability:.6f}"
        if result.exploitability != float("inf")
        else "inf"
    )
    line = (
        f"{commit}\t{expl_str}\t{result.iterations}\t"
        f"{result.wall_secs:.1f}\t{status}\t{description}\n"
    )
    with open(RESULTS_FILE, "a") as f:
        f.write(line)


def read_results() -> str:
    """Read results.tsv contents."""
    if RESULTS_FILE.exists():
        return RESULTS_FILE.read_text()
    return TSV_HEADER


def get_git_log(n: int = 20) -> str:
    """Get recent git log with diffs for the experiment file."""
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                f"-{n}",
                "--format=--- %h %s ---",
                "-p",
                "--",
                "crates/truco-solver/src/cfr_experiment.rs",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        log = result.stdout.strip()
        # Truncate if too long (keep under 8k chars to leave room in context)
        if len(log) > 8000:
            log = log[:8000] + "\n... (truncated)"
        return log
    except Exception as e:
        print(f"warning: git log unavailable: {e}", file=sys.stderr)
        return "(git log unavailable)"


def save_experiment_log(
    commit: str, result: ExperimentResult, description: str, run_output: str
):
    """Save the full experiment log to autoresearch/logs/<commit>.log."""
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"{commit}.log"
    with open(log_path, "w") as f:
        f.write(f"commit: {commit}\n")
        f.write(f"description: {description}\n")
        f.write(f"exploitability: {result.exploitability:.6f}\n")
        f.write(f"iterations: {result.iterations}\n")
        f.write(f"wall_secs: {result.wall_secs:.1f}\n")
        f.write(f"compile_ok: {result.compile_ok}\n")
        f.write(f"run_ok: {result.run_ok}\n")
        if result.error:
            f.write(f"error: {result.error}\n")
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Run output:\n{run_output}\n")


def save_experiment_snapshot(commit: str, description: str, code: str):
    """Save the proposed experiment source so failed/discarded runs are recoverable."""
    LOGS_DIR.mkdir(exist_ok=True)
    snapshot_path = LOGS_DIR / f"{commit}.rs"
    with open(snapshot_path, "w") as f:
        f.write(f"// commit: {commit}\n")
        f.write(f"// description: {description}\n\n")
        f.write(code)


def git_commit(message: str) -> str:
    """Create a git commit, return short hash."""
    subprocess.run(
        ["git", "add", "crates/truco-solver/src/cfr_experiment.rs"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git commit failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_revert():
    """Revert the last commit (discard experiment).

    Preserves results.tsv across the reset, since git reset --hard would
    otherwise wipe uncommitted changes to tracked files.
    """
    # Save results.tsv before reset — it's tracked but never committed
    # as part of experiment commits, so git reset --hard would destroy it.
    results_backup = RESULTS_FILE.read_text() if RESULTS_FILE.exists() else None
    subprocess.run(
        ["git", "reset", "--hard", "HEAD~1"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    # Restore results.tsv so the full experiment history is preserved
    if results_backup is not None:
        RESULTS_FILE.write_text(results_backup)


def get_best_exploitability() -> float:
    """Get the best exploitability from results.tsv."""
    best = float("inf")
    for line in read_results().strip().split("\n")[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) >= 5 and parts[4] == "keep":
            try:
                expl = float(parts[1])
                best = min(best, expl)
            except ValueError:
                pass
    return best


def short_head() -> str:
    """Current short commit hash, or '' if unavailable."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()


def candidate_params(config: LLMConfig, time_budget: int) -> dict:
    """The launch params logged with every candidate."""
    return {
        "provider": config.provider,
        "model": config.model,
        "time_budget_secs": time_budget,
    }


def run_baseline(
    time_budget: int, config: LLMConfig, tracker: Tracker
) -> ExperimentResult:
    """Run the baseline (current code, no modifications)."""
    log("\n" + "=" * 60)
    log("BASELINE RUN")
    log("=" * 60)
    log(f"Running experiment for {time_budget}s...")

    result = run_experiment(time_budget)

    if not result.compile_ok:
        log(f"ERROR: Baseline doesn't compile!\n{result.error}")
        sys.exit(1)

    if not result.run_ok:
        log(f"ERROR: Baseline crashed!\n{result.error}")
        sys.exit(1)

    log(f"Baseline exploitability: {result.exploitability:.6f}")
    log(f"Baseline iterations: {result.iterations}")
    log(f"Baseline wall time: {result.wall_secs:.1f}s")

    commit = short_head()
    append_result(commit, result, "keep", "baseline")
    tracker.log_candidate(
        commit=commit,
        status="baseline",
        description="baseline",
        result=result,
        params=candidate_params(config, time_budget),
    )
    return result


def run_loop(config: LLMConfig, time_budget: int, tracker: Tracker):
    """Main experiment loop. Runs until interrupted."""
    program_md = PROGRAM_MD.read_text()
    experiment_num = 0
    params = candidate_params(config, time_budget)

    while True:
        experiment_num += 1
        log(f"\n{'=' * 60}")
        log(f"EXPERIMENT #{experiment_num}  [{datetime.now().strftime('%H:%M:%S')}]")
        log(f"  provider={config.provider}  model={config.model}")
        log(f"{'=' * 60}")

        best_so_far = get_best_exploitability()
        log(f"Best exploitability so far: {best_so_far:.6f}")

        # 1. Ask LLM for a proposal
        current_code = read_experiment_file()
        results_tsv = read_results()
        git_history = get_git_log(20)

        log("Asking LLM for next experiment...")
        llm_stop = None
        llm_worker = None
        llm_start = 0.0
        try:
            llm_stop, llm_worker, llm_start = start_heartbeat("llm")
            proposal = propose_experiment(
                config,
                program_md,
                current_code,
                results_tsv,
                git_history=git_history,
            )
            stop_heartbeat("llm", llm_stop, llm_worker, llm_start)
        except Exception as e:
            if llm_stop is not None and not llm_stop.is_set():
                stop_heartbeat("llm", llm_stop, llm_worker, llm_start)
            log(f"LLM error: {e}")
            time.sleep(10)
            continue

        description = proposal["description"] or f"experiment-{experiment_num}"
        log(f"Proposal: {description}")
        log(f"Reasoning: {proposal['reasoning']}")

        if not proposal["code"]:
            log("ERROR: LLM returned empty code. Skipping.")
            continue

        # 2. Write the new code
        old_code = current_code
        parent_commit = short_head()
        write_experiment_file(proposal["code"])

        # 3. Commit
        try:
            commit = git_commit(f"experiment: {description}")
        except subprocess.CalledProcessError:
            log("Git commit failed. Restoring old code.")
            write_experiment_file(old_code)
            continue

        log(f"Committed: {commit}")
        save_experiment_snapshot(commit, description, proposal["code"])

        # 4. Run the experiment
        log(
            "Running experiment for "
            f"{time_budget}s ({format_duration(time_budget)} budget)..."
        )
        result = run_experiment(time_budget)

        # Common MLflow args for whichever terminal status this candidate hits.
        track_common = {
            "commit": commit,
            "description": description,
            "result": result,
            "parent_commit": parent_commit,
            "params": params,
            "artifacts": [
                str(LOGS_DIR / f"{commit}.log"),
                str(LOGS_DIR / f"{commit}.rs"),
            ],
        }

        # 5. Evaluate
        if not result.compile_ok:
            log(f"COMPILE ERROR:\n{result.error[:500]}")
            save_experiment_log(commit, result, description, result.error)
            append_result(commit, result, "crash", description)
            tracker.log_candidate(status="crash", **track_common)
            git_revert()
            log("Reverted.")
            continue

        if not result.run_ok:
            log(f"RUNTIME ERROR:\n{result.error[:500]}")
            save_experiment_log(commit, result, description, result.full_output)
            append_result(commit, result, "crash", description)
            tracker.log_candidate(status="crash", **track_common)
            git_revert()
            log("Reverted.")
            continue

        save_experiment_log(commit, result, description, result.full_output)
        log(f"Exploitability: {result.exploitability:.6f} (best: {best_so_far:.6f})")
        log(f"Iterations: {result.iterations} in {result.wall_secs:.1f}s")

        # 6. Keep or discard
        if result.exploitability < best_so_far:
            log(f"IMPROVEMENT! {best_so_far:.6f} → {result.exploitability:.6f}")
            append_result(commit, result, "keep", description)
            tracker.log_candidate(status="keep", **track_common)
        else:
            log("No improvement. Discarding.")
            append_result(commit, result, "discard", description)
            tracker.log_candidate(status="discard", **track_common)
            git_revert()


def main():
    parser = argparse.ArgumentParser(description="Truco CFR Autoresearch")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "LLM model override "
            "(default: AUTORESEARCH_MODEL env var or provider default)"
        ),
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=600,
        help="Time budget per experiment in seconds (default: 600)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the baseline run (use if results.tsv already has one)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking for this run (still writes results.tsv)",
    )
    args = parser.parse_args()

    # Detect LLM provider from env vars
    config = detect_config(model_override=args.model)
    log(f"Using {config.provider} ({config.model})")

    # MLflow tracking (durable, fail-safe). Defaults to a local file store; set
    # CFR_AUTORESEARCH_MLFLOW_TRACKING_URI / MLFLOW_TRACKING_URI for a server.
    campaign_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    tracker = (
        Tracker.disabled(campaign_id)
        if args.no_mlflow
        else Tracker.create(campaign_id=campaign_id)
    )

    init_results()

    # Run baseline if needed
    if not args.skip_baseline:
        lines = read_results().strip().split("\n")
        if len(lines) <= 1:  # only header
            run_baseline(args.budget, config, tracker)

    log("\nStarting experiment loop. Press Ctrl+C to stop.\n")
    try:
        run_loop(config, args.budget, tracker)
    except KeyboardInterrupt:
        log("\n\nStopped by user.")
        log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
