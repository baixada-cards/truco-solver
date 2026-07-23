"""Optional MLflow tracking for autoresearch.

Design rule: **MLflow is observability, never a hard dependency of the research
run.** If mlflow is not installed, the tracking URI is unreachable, or any log
call fails, every method here becomes a no-op and the experiment loop keeps
going. A flaky tracker must never lose us a research run (the old `results.tsv`
bug already cost us results once — durable logging is the point, not new
fragility).

Tracking URI resolution (first non-empty wins):

    MLFLOW_TRACKING_URI
    > CFR_AUTORESEARCH_MLFLOW_TRACKING_URI
    > file:<repo>/autoresearch/mlruns   (durable local store, default)

For a remote, auth-gated server, MLflow's own client reads
``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD`` from the environment.
This module does not handle credentials—it respects the process environment.
Resolve secret references at the process boundary rather than in this module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STORE = f"file:{_REPO_ROOT / 'autoresearch' / 'mlruns'}"
DEFAULT_EXPERIMENT = "truco-autoresearch"

# MLflow rejects over-long param/tag values; keep well under historical limits.
_MAX_VALUE_LEN = 250


def _info(message: str) -> None:
    print(f"[mlflow] {message}", flush=True)


def _warn(message: str) -> None:
    print(f"[mlflow] WARNING: {message}", file=sys.stderr, flush=True)


def _clip(value: object) -> str:
    text = str(value)
    return text if len(text) <= _MAX_VALUE_LEN else text[: _MAX_VALUE_LEN - 1] + "…"


def resolve_tracking_uri(env: Optional[dict] = None) -> str:
    import os

    e = env if env is not None else os.environ
    for key in ("MLFLOW_TRACKING_URI", "CFR_AUTORESEARCH_MLFLOW_TRACKING_URI"):
        val = (e.get(key) or "").strip()
        if val:
            return val
    return _DEFAULT_STORE


class Tracker:
    """Thin, fail-safe wrapper over the MLflow tracking client."""

    def __init__(
        self, mlflow_module, experiment: str, campaign_id: str, tracking_uri: str
    ):
        self._mlflow = mlflow_module
        self.experiment = experiment
        self.campaign_id = campaign_id
        self.tracking_uri = tracking_uri

    @property
    def enabled(self) -> bool:
        return self._mlflow is not None

    @classmethod
    def disabled(
        cls, campaign_id: str = "", experiment: str = DEFAULT_EXPERIMENT
    ) -> "Tracker":
        return cls(None, experiment, campaign_id, "")

    @classmethod
    def create(
        cls,
        *,
        campaign_id: str,
        experiment: str = DEFAULT_EXPERIMENT,
        env: Optional[dict] = None,
    ) -> "Tracker":
        """Build a tracker. Returns a disabled (no-op) tracker on any failure."""
        import os

        uri = resolve_tracking_uri(env)
        # Modern MLflow puts the local file store in "maintenance mode" and
        # refuses it unless this opt-out is set. The remote tracking server is
        # DB-backed, so this only affects the zero-config local fallback.
        if uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        try:
            import mlflow  # heavy import; only when tracking is wanted

            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
        except Exception as exc:  # not installed / unreachable / bad config
            _warn(
                f"tracking disabled (setup failed: {exc}). "
                f"The research run continues without MLflow. URI was {uri!r}."
            )
            return cls.disabled(campaign_id, experiment)
        _info(f"tracking to {uri} (experiment={experiment!r}, campaign={campaign_id})")
        return cls(mlflow, experiment, campaign_id, uri)

    def log_candidate(
        self,
        *,
        commit: str,
        status: str,
        description: str,
        result=None,
        parent_commit: Optional[str] = None,
        params: Optional[dict] = None,
        artifacts: Sequence[str] = (),
    ) -> None:
        """Log one candidate (baseline or proposal) as its own MLflow run.

        ``status`` is one of ``baseline`` / ``keep`` / ``discard`` / ``crash``.
        Never raises: on any error it disables tracking and returns.
        """
        if not self.enabled:
            return
        try:
            with self._mlflow.start_run(run_name=f"{status}:{commit}"):
                tags = {
                    "status": status,
                    "commit": commit,
                    "campaign_id": self.campaign_id,
                }
                if parent_commit:
                    tags["parent_commit"] = parent_commit
                self._mlflow.set_tags({k: _clip(v) for k, v in tags.items()})

                logged_params = {"description": _clip(description)}
                if parent_commit:
                    logged_params["parent_commit"] = parent_commit
                if params:
                    logged_params.update({k: _clip(v) for k, v in params.items()})
                self._mlflow.log_params(logged_params)

                metrics: dict[str, float] = {}
                if result is not None:
                    if result.exploitability != float("inf"):
                        metrics["exploitability"] = float(result.exploitability)
                    metrics["iterations"] = float(result.iterations)
                    metrics["wall_secs"] = float(result.wall_secs)
                    metrics["accepted"] = 1.0 if status == "keep" else 0.0
                if metrics:
                    self._mlflow.log_metrics(metrics)

                for artifact in artifacts:
                    if artifact and Path(artifact).exists():
                        self._mlflow.log_artifact(str(artifact))
        except Exception as exc:
            _warn(
                f"log failed for {commit} ({status}): {exc}; "
                "disabling tracking for the rest of this run."
            )
            self._mlflow = None
