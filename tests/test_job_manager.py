"""End-to-end coverage for TrainingJobManager (spawn-process jobs, queue polling, cancel/delete).

The manager is the only thing standing between the Streamlit UI and a subprocess, so every
test here drives a real spawned child: either a tiny classification job through the engine,
or a real idle process used to watch the cancel/delete contracts.
"""

import json
import multiprocessing
import re
import time

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.tracking.manager import JobStatus, TrainingJob, TrainingJobManager

NUMERIC_FEATURES = ("gene_expression", "cell_density")
CATEGORICAL_FEATURE = "tissue_type"
TARGET_COLUMN = "responds"
TOTAL_ROWS = 120
TRAIN_ROWS = 90

# Kept deliberately small: the whole job has to finish inside a pytest run, and two cheap
# sklearn models with 2 folds each is enough to exercise the manager end to end.
JOB_MODELS = ("logistic_regression", "decision_tree")

# The wait is bounded so a wedged child fails loudly with its own logs instead of hanging CI.
# A full run (spawn + engine import + 4 trials + MLflow logging) measures ~45s on this repo.
JOB_DEADLINE_SECONDS = 180.0
POLL_INTERVAL_SECONDS = 0.5

TERMINAL_STATUSES = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


# ──────────────────────────────────────────────
# Module-level helpers: Windows spawn re-imports this module in the child, so anything
# the child has to reach (the idle target, the data builders) lives here, not in a test body.
# ──────────────────────────────────────────────
def _make_synthetic_frame(rows: int = TOTAL_ROWS, seed: int = 17) -> pd.DataFrame:
    """Two numeric features plus one categorical feature predicting a balanced binary target."""
    rng = np.random.default_rng(seed)
    category_effects = {"loam": 1.1, "clay": -0.6, "sand": -0.2}
    numeric_a = rng.normal(0.0, 1.0, rows)
    numeric_b = rng.normal(0.0, 1.0, rows)
    category = rng.choice(sorted(category_effects), size=rows)
    signal = (
        1.8 * numeric_a
        - 1.2 * numeric_b
        + np.array([category_effects[c] for c in category])
        + rng.normal(0.0, 0.4, rows)
    )
    target = (signal > np.median(signal)).astype(int)
    frame = pd.DataFrame(
        {
            NUMERIC_FEATURES[0]: numeric_a,
            NUMERIC_FEATURES[1]: numeric_b,
            CATEGORICAL_FEATURE: category,
            TARGET_COLUMN: target,
        }
    )
    return frame


def _make_train_and_test_frames():
    """Stratification-free but balanced split: the target is deterministic, so a shuffle is enough."""
    shuffled = _make_synthetic_frame().sample(frac=1.0, random_state=3).reset_index(drop=True)
    return shuffled.iloc[:TRAIN_ROWS].copy(), shuffled.iloc[TRAIN_ROWS:].copy()


def _build_job_config(train_df, test_df, experiment_name: str = "JobManagerE2E") -> dict:
    """The exact config shape _training_worker reads (see debug_manager.py for the reference flow)."""
    return {
        "task": "classification",
        "target": TARGET_COLUMN,
        "train_df": train_df,
        "test_df": test_df,
        # 'custom' plus an explicit model list keeps the child out of xgboost/lightgbm/catboost;
        # the worker forwards config['selected_models'] straight to AutoMLTrainer.train.
        "preset": "custom",
        "n_trials": 2,
        "timeout": 30,
        "time_budget": 90,
        "selected_models": list(JOB_MODELS),
        "use_ensemble": False,
        "use_deep_learning": False,
        "ensemble_mode": "single",
        "experiment_name": experiment_name,
        "validation_strategy": "cv",
        "validation_params": {"folds": 2},
        "optimization_mode": "random",
        "optimization_metric": "accuracy",
        "target_metric_name": "ACCURACY",
        "random_state": 7,
        "early_stopping": 10,
        # The child resolves its own tracking URI (manager.py:116), so pin the store this test
        # process is already using; that is also what app.py:3372 does for real UI submissions.
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
    }


def _idle_child(seconds: float):
    """A real spawned child that only waits, used to watch cancel and delete."""
    time.sleep(seconds)


# ──────────────────────────────────────────────
# Manager polling helpers
# ──────────────────────────────────────────────
def _poll(manager: TrainingJobManager):
    """Poll once. poll_updates throttles itself to 0.5s, so reset the throttle first: the
    bounded loop below should wait on the child, not on the manager's own rate limiter."""
    manager._last_poll = 0.0
    manager.poll_updates()


def _describe_job(job) -> str:
    if job is None:
        return "job is missing from the manager"
    tail = "\n".join(job.logs[-40:]) if job.logs else "<no logs captured>"
    return (
        f"status={job.status!r} error_msg={job.error_msg!r} "
        f"best_score={job.best_score!r} mlflow_run_id={job.mlflow_run_id!r}\n"
        f"--- last job logs ---\n{tail}"
    )


def _await_terminal_status(manager: TrainingJobManager, job_id: str, deadline: float = JOB_DEADLINE_SECONDS):
    """Drain the manager the way every Streamlit rerun does until the job leaves the active states."""
    deadline_at = time.monotonic() + deadline
    while time.monotonic() < deadline_at:
        _poll(manager)
        job = manager.get_job(job_id)
        if job is not None and job.status in TERMINAL_STATUSES:
            # The child's log queue and status queue flush independently, so keep draining
            # after the done payload to pick up the trailing log lines the UI renders.
            for _ in range(5):
                time.sleep(POLL_INTERVAL_SECONDS)
                _poll(manager)
            return manager.get_job(job_id)
        time.sleep(POLL_INTERVAL_SECONDS)
    job = manager.get_job(job_id)
    pytest.fail(
        f"Job {job_id} never reached a terminal status within {deadline:.0f}s.\n{_describe_job(job)}"
    )


def _poll_until(manager: TrainingJobManager, job, predicate, what: str, timeout: float = 15.0):
    """Queue payloads reach the parent asynchronously, so poll until they land (with a ceiling)."""
    deadline_at = time.monotonic() + timeout
    while time.monotonic() < deadline_at:
        _poll(manager)
        if predicate(job):
            return job
        time.sleep(0.05)
    pytest.fail(f"{what} did not reach the manager within {timeout:.1f}s.\n{_describe_job(job)}")


def _make_job_with_started_child(manager: TrainingJobManager, job_id: str = "idle-job"):
    """Attach a genuinely running (but idle) spawned child to the manager as a job."""
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=_idle_child, args=(120.0,), daemon=True)
    process.start()
    job = TrainingJob(
        job_id=job_id,
        name=f"idle_{job_id}",
        config={},
        status=JobStatus.RUNNING,
        _process=process,
        _log_queue=ctx.Queue(),
        _status_queue=ctx.Queue(),
        _pause_event=ctx.Event(),
    )
    manager.jobs[job_id] = job
    return job, process


@pytest.fixture
def idle_child_job():
    """A manager holding one live spawned child; the child is terminated on teardown."""
    manager = TrainingJobManager()
    job, process = _make_job_with_started_child(manager)
    try:
        yield manager, job
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def _make_queued_job(manager: TrainingJobManager, job_id: str, logs=(), updates=()):
    """A job whose queues already hold the messages a child would have pushed."""
    ctx = multiprocessing.get_context("spawn")
    log_queue, status_queue = ctx.Queue(), ctx.Queue()
    for line in logs:
        log_queue.put(("log", line))
    for update in updates:
        status_queue.put(update)
    job = TrainingJob(
        job_id=job_id,
        name=f"queued_{job_id}",
        config={},
        status=JobStatus.RUNNING,
        _log_queue=log_queue,
        _status_queue=status_queue,
        _pause_event=ctx.Event(),
    )
    manager.jobs[job_id] = job
    return job


# ──────────────────────────────────────────────
# Synthetic data sanity (the job under test trains on it)
# ──────────────────────────────────────────────
def test_synthetic_frame_is_balanced_with_the_expected_columns():
    frame = _make_synthetic_frame()
    assert list(frame.columns) == [*NUMERIC_FEATURES, CATEGORICAL_FEATURE, TARGET_COLUMN]
    assert len(frame) == TOTAL_ROWS
    counts = frame[TARGET_COLUMN].value_counts()
    assert counts.nunique() == 1, f"target must be balanced, got {counts.to_dict()}"
    assert set(frame[CATEGORICAL_FEATURE]) == {"clay", "loam", "sand"}

    train_df, test_df = _make_train_and_test_frames()
    assert len(train_df) == TRAIN_ROWS and len(test_df) == TOTAL_ROWS - TRAIN_ROWS
    assert list(train_df.columns) == list(test_df.columns)
    assert train_df[TARGET_COLUMN].nunique() == test_df[TARGET_COLUMN].nunique() == 2


def test_get_job_and_deleting_an_unknown_job_are_safe():
    manager = TrainingJobManager()
    assert manager.get_job("no-such-job") is None
    assert manager.list_jobs() == []
    assert manager.has_running_jobs() is False
    assert manager.active_count() == 0

    _poll(manager)
    manager.delete_job("no-such-job")
    manager.delete_job("no-such-job", delete_mlflow_run=True)
    assert manager.get_job("no-such-job") is None


# ──────────────────────────────────────────────
# Queue protocol consumed by poll_updates
# ──────────────────────────────────────────────
def test_poll_updates_maps_child_queue_messages_onto_the_job():
    manager = TrainingJobManager()
    trial = {"Global Trial": 1, "Model": "logistic_regression", "ACCURACY": 0.83}
    job = _make_queued_job(
        manager,
        "protocol-job",
        logs=["[JOB] Starting preprocessing for experiment: JobManagerE2E",
              "[JOB] Preprocessing done. Features: 5"],
        updates=[
            {"type": "trial", "trial": trial, "score": 0.83,
             "full_name": "logistic_regression - Trial 1"},
            {"type": "trial", "trial": {**trial, "Global Trial": 2}, "score": 0.91,
             "full_name": "logistic_regression - Trial 2"},
            {"type": "report", "model_name": "logistic_regression",
             "report": {"score": 0.91, "plots": {"roc": ("pil", b"png-bytes")}}},
            {"type": "done", "best_score": 0.91,
             "best_params": {"model_name": "logistic_regression", "C": 1.0},
             "mlflow_run_id": "deadbeef" * 4, "mlflow_experiment": "JobManagerE2E",
             "model_summaries": {"logistic_regression": {"score": 0.91, "trial_name": "t"}},
             "eval_metrics": {"accuracy": 0.88}, "consumption_code": "mlflow.sklearn.load_model(...)"},
        ],
    )

    _poll_until(
        manager,
        job,
        lambda seen: seen.status == JobStatus.COMPLETED and len(seen.logs) == 2,
        "the completion payload and the child logs",
    )

    assert job.logs == ["[JOB] Starting preprocessing for experiment: JobManagerE2E",
                        "[JOB] Preprocessing done. Features: 5"]
    assert job.trials_data == [trial, {**trial, "Global Trial": 2}]
    assert job.best_score == pytest.approx(0.91)
    assert job.report_data["logistic_regression"]["plots"]["roc"] == ("pil", b"png-bytes")
    assert job.status == JobStatus.COMPLETED
    assert job.end_time is not None
    assert job.mlflow_run_id == "deadbeef" * 4
    assert job.mlflow_experiment == "JobManagerE2E"
    assert job.model_summaries["logistic_regression"]["score"] == pytest.approx(0.91)
    # The completion payload lands inside job.config, which is where the Results tab reads it.
    assert job.config["best_params"]["model_name"] == "logistic_regression"
    assert job.config["eval_metrics"] == {"accuracy": 0.88}
    assert job.config["consumption_code"].startswith("mlflow.sklearn")
    assert manager.has_running_jobs() is False


def test_poll_updates_records_a_worker_error_and_caps_the_log_list():
    manager = TrainingJobManager()
    failed = _make_queued_job(
        manager,
        "error-job",
        logs=["[JOB ERROR] target column missing"],
        updates=[{"type": "error", "error": "KeyError: 'responds'"}],
    )
    flooded = _make_queued_job(
        manager,
        "flood-job",
        logs=[f"line {i}" for i in range(520)],
    )

    _poll_until(manager, failed, lambda seen: seen.status == JobStatus.FAILED, "the error payload")
    _poll_until(
        manager,
        flooded,
        lambda seen: bool(seen.logs) and seen.logs[-1] == "line 519",
        "the whole flood of logs",
    )
    assert failed.error_msg == "KeyError: 'responds'"
    assert failed.end_time is not None

    assert len(flooded.logs) == 500, "job.logs is documented to cap at the newest 500 entries"
    assert flooded.logs[-1] == "line 519"


def test_pause_and_resume_move_the_job_state_and_the_child_event():
    manager = TrainingJobManager()
    job = _make_queued_job(manager, "pause-job")

    manager.pause_job(job.job_id)
    assert job.status == JobStatus.PAUSED
    assert job._pause_event.is_set()
    assert manager.has_running_jobs() is False, "a paused job is active but not running"
    assert manager.active_count() == 1

    manager.resume_job(job.job_id)
    assert job.status == JobStatus.RUNNING
    assert not job._pause_event.is_set()
    assert manager.has_running_jobs() is True

    # Unknown ids must not raise, and pausing a finished job is a no-op.
    manager.pause_job("no-such-job")
    manager.resume_job("no-such-job")
    job.status = JobStatus.COMPLETED
    manager.pause_job(job.job_id)
    assert job.status == JobStatus.COMPLETED


# ──────────────────────────────────────────────
# Cancel / delete against a real child process
# ──────────────────────────────────────────────
def test_cancel_job_terminates_the_child_and_marks_it_cancelled(idle_child_job):
    manager, job = idle_child_job
    assert manager.has_running_jobs() is True

    manager.cancel_job(job.job_id)

    assert job.status == JobStatus.CANCELLED
    assert job.end_time is not None
    assert not job._process.is_alive()
    assert manager.has_running_jobs() is False
    assert manager.active_count() == 0

    # Cancelling twice, or cancelling an unknown id, must not raise.
    manager.cancel_job(job.job_id)
    manager.cancel_job("no-such-job")
    assert job.status == JobStatus.CANCELLED


def test_delete_job_terminates_a_running_child_before_removing_it():
    manager = TrainingJobManager()
    job, process = _make_job_with_started_child(manager, job_id="delete-me")
    try:
        assert manager.get_job("delete-me") is job

        manager.delete_job("delete-me")

        assert manager.get_job("delete-me") is None
        assert "delete-me" not in manager.jobs
        assert manager.list_jobs() == []
        assert not job._process.is_alive(), "deleting a live job has to stop its child"
        assert job.status == JobStatus.CANCELLED
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


# ──────────────────────────────────────────────
# The real thing: a spawned job trained through the engine
# ──────────────────────────────────────────────
def test_real_job_completes_and_delivers_everything_the_ui_renders(tmp_path, monkeypatch):
    # The child writes its whitebox notebook and model cards relative to its own cwd.
    monkeypatch.chdir(tmp_path)
    train_df, test_df = _make_train_and_test_frames()
    dataset_path = tmp_path / "train.csv"
    train_df.to_csv(dataset_path, index=False)

    manager = TrainingJobManager()
    config = _build_job_config(train_df, test_df)
    config["dataset_path"] = str(dataset_path)
    job_id = manager.submit_job(config, name="JobManagerE2E")

    assert manager.has_running_jobs() is True, "a freshly submitted job must read as running"
    assert re.fullmatch(r"[0-9a-f]{8}", job_id), f"unexpected job id format: {job_id}"

    job = _await_terminal_status(manager, job_id)
    assert job.status == JobStatus.COMPLETED, (
        f"the job did not complete.\n{_describe_job(job)}"
    )
    assert manager.has_running_jobs() is False, "the UI must stop refreshing once the child is done"
    assert manager.active_count() == 0
    assert job.end_time >= job.start_time
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2}", job.duration_str)

    best_params = job.config.get("best_params")
    assert isinstance(best_params, dict) and best_params, _describe_job(job)
    assert best_params.get("model_name") in JOB_MODELS, f"unexpected best model: {best_params}"

    assert isinstance(job.best_score, float) and 0.0 <= job.best_score <= 1.0, _describe_job(job)

    eval_metrics = job.config.get("eval_metrics")
    assert isinstance(eval_metrics, dict) and eval_metrics, _describe_job(job)
    assert "accuracy" in eval_metrics and 0.0 <= eval_metrics["accuracy"] <= 1.0

    assert job.mlflow_run_id, _describe_job(job)
    assert job.mlflow_experiment == "JobManagerE2E"
    run = mlflow.get_run(job.mlflow_run_id)
    assert run.info.experiment_id
    assert run.info.status == "FINISHED"

    assert job.trials_data, _describe_job(job)
    assert {"Model", "Identifier", "Duration (s)", "ACCURACY"} <= set(job.trials_data[0])
    assert {row["Model"] for row in job.trials_data} <= set(JOB_MODELS)

    assert job.model_summaries, _describe_job(job)
    assert set(job.model_summaries) <= set(JOB_MODELS)
    for summary in job.model_summaries.values():
        assert 0.0 <= summary["score"] <= 1.0
        assert summary["trial_name"]

    assert job.report_data, _describe_job(job)
    for report in job.report_data.values():
        assert isinstance(report["plots"], dict)
        for payload in report["plots"].values():
            kind, buffer = payload
            assert kind in ("pil", "mpl") and isinstance(buffer, bytes) and buffer

    # The Results tab renders the worker's log list, so progress has to reach the parent.
    assert job.logs, _describe_job(job)
    assert any("[JOB] Preprocessing done" in line for line in job.logs), _describe_job(job)
    assert any("Training complete" in line for line in job.logs), _describe_job(job)
    progress_lines = [line for line in job.logs if re.search(r"Trial|optimization", line, re.I)]
    assert progress_lines, f"no training progress reached the job log.\n{_describe_job(job)}"
    assert any("model_card" in summary.get("metrics", {}) for summary in job.model_summaries.values())

    # The Results/Overview tabs json.dumps these payloads; report_data is deliberately excluded
    # because its plots travel as (kind, png-bytes) buffers for st.image.
    for label, payload in (("model_summaries", job.model_summaries),
                           ("eval_metrics", eval_metrics),
                           ("trials_data", job.trials_data)):
        try:
            json.dumps(payload)
        except TypeError as serialisation_error:
            pytest.fail(
                f"job.{label} is not JSON-serializable for the UI: {serialisation_error}.\n"
                f"{_describe_job(job)}"
            )

    run_id = job.mlflow_run_id
    manager.delete_job(job_id, delete_mlflow_run=True)

    assert manager.get_job(job_id) is None
    assert manager.jobs == {}
    assert mlflow.get_run(run_id).info.lifecycle_stage == "deleted"


def test_a_worker_that_raises_surfaces_a_failed_job_and_the_traceback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_df, _ = _make_train_and_test_frames()
    manager = TrainingJobManager()
    config = _build_job_config(train_df, None)
    # A target the child cannot find makes preprocessing raise on purpose.
    config["target"] = "not_a_column"

    job_id = manager.submit_job(config, name="JobManagerFailure")
    job = _await_terminal_status(manager, job_id)

    assert job.status == JobStatus.FAILED, _describe_job(job)
    assert job.error_msg
    assert manager.has_running_jobs() is False
    assert any("[JOB ERROR]" in line for line in job.logs), _describe_job(job)
    assert any("Traceback (most recent call last)" in line for line in job.logs), _describe_job(job)
