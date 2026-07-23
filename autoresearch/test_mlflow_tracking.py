"""Tests for the fail-safe MLflow tracker (run from the autoresearch dir)."""

from __future__ import annotations

from types import SimpleNamespace

import mlflow_tracking
from mlflow_tracking import Tracker, resolve_tracking_uri


def _result(expl=0.12, iters=40, wall=600.0):
    return SimpleNamespace(exploitability=expl, iterations=iters, wall_secs=wall)


def test_uri_precedence(tmp_path):
    assert resolve_tracking_uri({"MLFLOW_TRACKING_URI": "http://a"}) == "http://a"
    assert (
        resolve_tracking_uri({"CFR_AUTORESEARCH_MLFLOW_TRACKING_URI": "http://b"})
        == "http://b"
    )
    # MLFLOW_TRACKING_URI wins over the project-specific var.
    assert (
        resolve_tracking_uri(
            {
                "MLFLOW_TRACKING_URI": "http://a",
                "CFR_AUTORESEARCH_MLFLOW_TRACKING_URI": "http://b",
            }
        )
        == "http://a"
    )
    assert resolve_tracking_uri({}).startswith("file:")


def test_disabled_tracker_is_noop():
    t = Tracker.disabled("camp")
    assert t.enabled is False
    # Must not raise.
    t.log_candidate(commit="abc", status="keep", description="x", result=_result())


def test_happy_path_logs_to_file_store(tmp_path):
    uri = f"file:{tmp_path / 'mlruns'}"
    t = Tracker.create(
        campaign_id="camp1",
        experiment="truco-autoresearch-test",
        env={"CFR_AUTORESEARCH_MLFLOW_TRACKING_URI": uri},
    )
    assert t.enabled is True
    t.log_candidate(
        commit="deadbee",
        status="keep",
        description="DCFR+ test candidate",
        result=_result(expl=0.0975, iters=55),
        parent_commit="cafe123",
        params={"provider": "anthropic", "model": "claude-x", "time_budget_secs": 600},
    )

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("truco-autoresearch-test")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    data = runs[0].data
    assert abs(data.metrics["exploitability"] - 0.0975) < 1e-9
    assert data.metrics["accepted"] == 1.0
    assert data.tags["status"] == "keep"
    assert data.tags["campaign_id"] == "camp1"


def test_log_failure_disables_tracking_without_raising():
    class BoomMlflow:
        def start_run(self, *a, **k):
            raise RuntimeError("tracking server down")

    t = Tracker(BoomMlflow(), "exp", "camp", "http://unreachable")
    assert t.enabled is True
    # Must swallow the error and disable, never propagate.
    t.log_candidate(commit="abc", status="keep", description="x", result=_result())
    assert t.enabled is False


def test_inf_exploitability_is_not_logged(tmp_path):
    uri = f"file:{tmp_path / 'mlruns'}"
    t = Tracker.create(
        campaign_id="camp2",
        experiment="truco-autoresearch-crash",
        env={"CFR_AUTORESEARCH_MLFLOW_TRACKING_URI": uri},
    )
    t.log_candidate(
        commit="badc0de",
        status="crash",
        description="compile error",
        result=_result(expl=float("inf"), iters=0, wall=0.0),
    )
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("truco-autoresearch-crash")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    # inf exploitability is never logged as a metric.
    assert "exploitability" not in runs[0].data.metrics
    assert runs[0].data.tags["status"] == "crash"


assert mlflow_tracking  # keep import referenced
