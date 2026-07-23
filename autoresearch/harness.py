"""Harness for running and evaluating CFR experiments.

Compiles the solver, runs it for a fixed time budget, and extracts the
exploitability metric.
"""

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure cargo is in PATH (rustup installs to ~/.cargo/bin)
cargo_bin = Path.home() / ".cargo" / "bin"
if cargo_bin.exists() and str(cargo_bin) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"

# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOLVER_CRATE = PROJECT_ROOT / "crates" / "truco-solver"
EXPERIMENT_FILE = SOLVER_CRATE / "src" / "cfr_experiment.rs"
EXPERIMENT_BIN = "experiment"
DEFAULT_HEARTBEAT_SECS = 5.0
HEARTBEAT_BAR_WIDTH = 24


@dataclass
class ExperimentResult:
    exploitability: float
    iterations: int
    wall_secs: float
    compile_ok: bool
    run_ok: bool
    error: str = ""
    full_output: str = ""


def format_duration(seconds: float) -> str:
    """Format a duration as MM:SS or HH:MM:SS."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _heartbeat_interval_secs() -> float:
    """Heartbeat cadence; overridable for live tmux sessions."""
    raw = os.environ.get("AUTORESEARCH_HEARTBEAT_SECS", str(DEFAULT_HEARTBEAT_SECS))
    try:
        return max(0.5, float(raw))
    except ValueError:
        return DEFAULT_HEARTBEAT_SECS


def _known_total_bar(elapsed: float, total: float) -> str:
    ratio = 0.0 if total <= 0 else min(1.0, elapsed / total)
    filled = int(ratio * HEARTBEAT_BAR_WIDTH)
    return "[" + "#" * filled + "-" * (HEARTBEAT_BAR_WIDTH - filled) + "]"


def _unknown_total_bar(elapsed: float) -> str:
    if HEARTBEAT_BAR_WIDTH <= 1:
        return "[#]"

    cycle = (HEARTBEAT_BAR_WIDTH - 1) * 2
    pos = int(elapsed / _heartbeat_interval_secs()) % cycle
    if pos >= HEARTBEAT_BAR_WIDTH:
        pos = cycle - pos

    chars = ["-"] * HEARTBEAT_BAR_WIDTH
    chars[pos] = "#"
    return "[" + "".join(chars) + "]"


def _heartbeat_thread(
    stop_event: threading.Event,
    label: str,
    start: float,
    total_secs: float | None = None,
) -> None:
    """Print periodic progress so long-running phases never look frozen."""
    interval = _heartbeat_interval_secs()
    while not stop_event.wait(interval):
        elapsed = time.time() - start
        timestamp = time.strftime("%H:%M:%S")
        if total_secs is None:
            bar = _unknown_total_bar(elapsed)
            detail = f"{format_duration(elapsed)} elapsed"
        else:
            remaining = max(0.0, total_secs - elapsed)
            bar = _known_total_bar(elapsed, total_secs)
            if remaining > 0:
                detail = (
                    f"{format_duration(elapsed)} elapsed, "
                    f"{format_duration(remaining)} left"
                )
            else:
                detail = f"{format_duration(elapsed)} elapsed, finishing up"

        print(f"[{timestamp}] [{label:<8}] {bar} {detail}", flush=True)


def start_heartbeat(
    label: str,
    total_secs: float | None = None,
) -> tuple[threading.Event, threading.Thread, float]:
    """Start a background heartbeat for a long-running phase."""
    start = time.time()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_heartbeat_thread,
        args=(stop_event, label, start, total_secs),
        daemon=True,
    )
    worker.start()
    return stop_event, worker, start


def stop_heartbeat(
    label: str,
    stop_event: threading.Event,
    worker: threading.Thread,
    start: float,
) -> float:
    """Stop a heartbeat and print a concise completion line."""
    stop_event.set()
    worker.join(timeout=0.2)
    elapsed = time.time() - start
    timestamp = time.strftime("%H:%M:%S")
    print(
        f"[{timestamp}] [{label:<8}] completed in {format_duration(elapsed)}",
        flush=True,
    )
    return elapsed


def compile_solver() -> tuple[bool, str]:
    """Compile the solver in release mode. Returns (success, error_msg)."""
    output_lines: list[str] = []
    stop_heartbeat_event, heartbeat_worker, heartbeat_start = start_heartbeat("compile")

    try:
        proc = subprocess.Popen(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "truco-solver",
                "--bin",
                EXPERIMENT_BIN,
            ],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            output_lines.append(line)
            print(line, end="", flush=True)
        returncode = proc.wait()
    finally:
        stop_heartbeat(
            "compile", stop_heartbeat_event, heartbeat_worker, heartbeat_start
        )

    if returncode != 0:
        return False, "".join(output_lines)[-2000:]
    return True, ""


def run_experiment(time_budget_secs: int = 600) -> ExperimentResult:
    """Compile and run one experiment. Streams output live. Returns the result."""

    # Compile
    ok, err = compile_solver()
    if not ok:
        return ExperimentResult(
            exploitability=float("inf"),
            iterations=0,
            wall_secs=0,
            compile_ok=False,
            run_ok=False,
            error=err[-2000:],  # last 2000 chars of compile error
        )

    # Run the experiment binary and stream stdout+stderr live so the terminal
    # never looks silent.
    bin_path = PROJECT_ROOT / "target" / "release" / EXPERIMENT_BIN
    env = os.environ.copy()
    env["RUST_LOG"] = "info"
    env["EXPERIMENT_TIME_BUDGET"] = str(time_budget_secs)

    stop_clock, ticker, start = start_heartbeat("run", total_secs=time_budget_secs)

    output_lines: list[str] = []
    returncode = 0
    try:
        proc = subprocess.Popen(
            [str(bin_path)],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr so progress bars appear
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            output_lines.append(line)
            print(line, end="", flush=True)
        proc.wait(timeout=time_budget_secs + 600)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        stop_heartbeat("run", stop_clock, ticker, start)
        return ExperimentResult(
            exploitability=float("inf"),
            iterations=0,
            wall_secs=time.time() - start,
            compile_ok=True,
            run_ok=False,
            error="Timed out",
        )
    finally:
        if not stop_clock.is_set():
            stop_heartbeat("run", stop_clock, ticker, start)

    wall_secs = time.time() - start
    output = "".join(output_lines)

    if returncode != 0:
        return ExperimentResult(
            exploitability=float("inf"),
            iterations=0,
            wall_secs=wall_secs,
            compile_ok=True,
            run_ok=False,
            error=output[-2000:],
        )

    # Parse output
    expl = _extract_float(output, r"exploitability:\s*([\d.]+)")
    iters = _extract_int(output, r"iterations:\s*(\d+)")

    return ExperimentResult(
        exploitability=expl if expl is not None else float("inf"),
        iterations=iters or 0,
        wall_secs=wall_secs,
        compile_ok=True,
        run_ok=True,
        full_output=output,
    )


def read_experiment_file() -> str:
    """Read the current contents of cfr_experiment.rs."""
    return EXPERIMENT_FILE.read_text()


def write_experiment_file(code: str) -> None:
    """Write new contents to cfr_experiment.rs."""
    EXPERIMENT_FILE.write_text(code)


def _extract_float(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def _extract_int(text: str, pattern: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None
