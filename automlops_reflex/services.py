from __future__ import annotations

from datetime import datetime
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any
import json
import os

import mlflow
import pandas as pd
import requests

from src.core.drift import DriftDetector
from src.core.processor import AutoMLDataProcessor
from src.core.data_lake import DataLake
from src.deploy.hf_deploy import deploy_to_huggingface
from src.engines.classical import AutoMLTrainer
from src.engines.stability import StabilityAnalyzer
from src.engines.vision import CVAutoMLTrainer
from src.tracking.manager import TrainingJobManager, JobStatus
from src.tracking.mlflow import (
    get_all_runs,
    get_registered_models,
    get_model_details,
    get_run_details,
    load_registered_model,
    register_model_from_run,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / ".runtime_logs"

SERVICE_SPECS: dict[str, dict[str, Any]] = {
    "api": {
        "label": "Serving API",
        "url": "http://127.0.0.1:8000/docs",
        "port": 8000,
        "command": [
            sys.executable,
            "-m",
            "uvicorn",
            "api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
    },
    "mlflow": {
        "label": "MLflow Tracking",
        "url": "http://127.0.0.1:5000",
        "port": 5000,
        "command": [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
            "--backend-store-uri",
            "./mlruns",
        ],
    },
}

_PROCESS_REGISTRY: dict[str, subprocess.Popen[Any]] = {}
_JOB_MANAGER = TrainingJobManager()

# These keys are trainable search strategies or ensemble wrappers, not base models to expose
# in the manual model picker.
NON_BASE_MODEL_KEYS = frozenset(
    {
        "voting_ensemble",
        "custom_voting",
        "custom_stacking",
        "custom_bagging",
        "bagging",
        "stacking_ensemble",
    }
)


def _root(project_root: str | Path | None = None) -> Path:
    return Path(project_root) if project_root else PROJECT_ROOT


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def _format_timestamp(value: Any) -> str:
    if value is None or value == "":
        return "Not available"
    try:
        stamp = pd.to_datetime(value)
        if pd.isna(stamp):
            return "Not available"
        return stamp.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value)


def _stringify_schema_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _humanize_model_name(model_name: str) -> str:
    return model_name.replace("_", " ").replace("-", " ").title()


def _summarize_metrics(frame: pd.DataFrame, row_index: int) -> str:
    metric_columns = [column for column in frame.columns if column.startswith("metrics.")]
    parts: list[str] = []
    for column in metric_columns[:3]:
        value = frame.iloc[row_index][column]
        if pd.isna(value):
            continue
        try:
            parts.append(f"{column.replace('metrics.', '')}: {float(value):.4f}")
        except Exception:
            parts.append(f"{column.replace('metrics.', '')}: {value}")
    return ", ".join(parts) if parts else "No metrics logged"


def _safe_get_all_runs() -> pd.DataFrame:
    """Return runs using current URI; fallback to local file store if legacy sqlite metadata fails."""
    try:
        runs = get_all_runs()
        if isinstance(runs, pd.DataFrame):
            return runs
        return pd.DataFrame()
    except Exception:
        pass

    previous_uri = mlflow.get_tracking_uri()
    try:
        fallback_uri = (PROJECT_ROOT / "mlruns").resolve().as_uri()
        mlflow.set_tracking_uri(fallback_uri)
        runs = get_all_runs()
        if isinstance(runs, pd.DataFrame):
            return runs
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    finally:
        try:
            mlflow.set_tracking_uri(previous_uri)
        except Exception:
            pass


def get_tracking_status() -> dict[str, str | bool]:
    uri = mlflow.get_tracking_uri()
    is_dagshub = "dagshub.com" in uri.lower()
    return {
        "uri": uri,
        "is_dagshub": is_dagshub,
        "status_label": "Connected to DagsHub" if is_dagshub else "Local MLflow",
    }


def connect_dagshub(username: str, repo: str, token: str) -> dict[str, str | bool]:
    user = username.strip()
    repo_name = repo.strip()
    auth_token = token.strip()
    if not user or not repo_name or not auth_token:
        raise ValueError("Fill username, repository, and token.")

    remote_uri = f"https://dagshub.com/{user}/{repo_name}.mlflow"
    previous_uri = mlflow.get_tracking_uri()
    previous_user = os.environ.get("MLFLOW_TRACKING_USERNAME")
    previous_pass = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    previous_env_uri = os.environ.get("MLFLOW_TRACKING_URI")

    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = auth_token
        os.environ["MLFLOW_TRACKING_URI"] = remote_uri
        mlflow.set_tracking_uri(remote_uri)
        mlflow.search_experiments(max_results=1)
        return get_tracking_status()
    except Exception:
        if previous_user is None:
            os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
        else:
            os.environ["MLFLOW_TRACKING_USERNAME"] = previous_user

        if previous_pass is None:
            os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
        else:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = previous_pass

        if previous_env_uri is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = previous_env_uri

        mlflow.set_tracking_uri(previous_uri)
        raise


def disconnect_dagshub(local_uri: str = "sqlite:///mlflow.db") -> dict[str, str | bool]:
    mlflow.set_tracking_uri(local_uri)
    os.environ["MLFLOW_TRACKING_URI"] = local_uri
    os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
    os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
    return get_tracking_status()


def list_service_statuses() -> dict[str, dict[str, str | bool]]:
    statuses: dict[str, dict[str, str | bool]] = {}
    for name, spec in SERVICE_SPECS.items():
        process = _PROCESS_REGISTRY.get(name)
        if process and process.poll() is not None:
            _PROCESS_REGISTRY.pop(name, None)
            process = None

        online = _port_open(spec["port"])
        if online:
            status = "online"
            status_label = "Online"
        elif process is not None:
            status = "starting"
            status_label = "Starting"
        else:
            status = "stopped"
            status_label = "Stopped"

        statuses[name] = {
            "name": name,
            "label": spec["label"],
            "url": spec["url"],
            "status": status,
            "status_label": status_label,
            "pid": str(process.pid) if process is not None and process.poll() is None else "-",
            "ready": online,
        }
    return statuses


def start_service(name: str, project_root: str | Path | None = None, timeout: float = 12.0) -> dict[str, str | bool]:
    if name not in SERVICE_SPECS:
        raise ValueError(f"Unknown service: {name}")

    statuses = list_service_statuses()
    if statuses[name]["ready"]:
        return statuses[name]

    root = _root(project_root)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    log_handle = log_path.open("a", encoding="utf-8")

    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    process = subprocess.Popen(
        SERVICE_SPECS[name]["command"],
        cwd=root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    _PROCESS_REGISTRY[name] = process

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(SERVICE_SPECS[name]["port"]):
            break
        if process.poll() is not None:
            break
        time.sleep(0.5)

    return list_service_statuses()[name]


def stop_service(name: str) -> dict[str, str | bool]:
    process = _PROCESS_REGISTRY.get(name)
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    _PROCESS_REGISTRY.pop(name, None)
    return list_service_statuses()[name]


def summarize_project(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(project_root)
    lake = DataLake(root / "data_lake")

    datasets: list[dict[str, str]] = []
    total_versions = 0
    for dataset_name in lake.list_datasets():
        versions = lake.list_versions(dataset_name)
        total_versions += len(versions)
        latest_version = versions[0] if versions else "-"
        latest_updated = "Not available"
        preview_columns = "No preview"
        if versions:
            info = lake.get_version_info(dataset_name, latest_version)
            latest_updated = info.modified_at.strftime("%Y-%m-%d %H:%M")
            try:
                preview_frame = lake.load_version(dataset_name, latest_version, nrows=3)
                preview_columns = ", ".join(list(preview_frame.columns[:4])) or "No columns"
            except Exception:
                preview_columns = "Binary or unsupported preview"
        datasets.append(
            {
                "name": dataset_name,
                "versions": str(len(versions)),
                "latest_version": latest_version,
                "latest_updated": latest_updated,
                "preview_columns": preview_columns,
            }
        )

    telemetry_path = root / "data_lake" / "monitoring" / "api_telemetry.csv"
    telemetry_rows = 0
    telemetry_last_seen = "No telemetry yet"
    telemetry_columns = "No telemetry columns"
    if telemetry_path.exists():
        try:
            telemetry_frame = pd.read_csv(telemetry_path)
            telemetry_rows = len(telemetry_frame)
            telemetry_columns = ", ".join(list(telemetry_frame.columns[:6])) or "No telemetry columns"
            if "__timestamp" in telemetry_frame.columns and not telemetry_frame.empty:
                telemetry_last_seen = _format_timestamp(telemetry_frame["__timestamp"].iloc[-1])
        except Exception:
            telemetry_last_seen = "Telemetry file unreadable"

    models: list[dict[str, str]] = []
    try:
        registered_models = get_registered_models()
    except Exception:
        registered_models = []
    for model in registered_models[:8]:
        latest_version = "-"
        latest_stage = "Unassigned"
        latest_versions = getattr(model, "latest_versions", []) or []
        if latest_versions:
            latest_version = str(getattr(latest_versions[0], "version", "-"))
            latest_stage = str(getattr(latest_versions[0], "current_stage", "Unassigned") or "Unassigned")
        models.append(
            {
                "name": str(getattr(model, "name", "Unnamed model")),
                "version": latest_version,
                "stage": latest_stage,
                "description": str(getattr(model, "description", "") or "No description provided"),
            }
        )

    recent_runs: list[dict[str, str]] = []
    runs_frame = _safe_get_all_runs()
    if isinstance(runs_frame, pd.DataFrame) and not runs_frame.empty:
        if "start_time" in runs_frame.columns:
            runs_frame = runs_frame.sort_values("start_time", ascending=False)
        for row_index in range(min(len(runs_frame), 8)):
            recent_runs.append(
                {
                    "run_id": str(runs_frame.iloc[row_index].get("run_id", "")),
                    "experiment": str(runs_frame.iloc[row_index].get("experiment_name", "Default")),
                    "status": str(runs_frame.iloc[row_index].get("status", "unknown")),
                    "started": _format_timestamp(runs_frame.iloc[row_index].get("start_time")),
                    "metrics": _summarize_metrics(runs_frame, row_index),
                }
            )

    return {
        "dataset_count": len(datasets),
        "dataset_version_count": total_versions,
        "registered_model_count": len(models),
        "run_count": len(runs_frame) if isinstance(runs_frame, pd.DataFrame) else 0,
        "telemetry_row_count": telemetry_rows,
        "telemetry_last_seen": telemetry_last_seen,
        "telemetry_columns": telemetry_columns,
        "datasets": datasets,
        "models": models,
        "runs": recent_runs,
        "service_statuses": list_service_statuses(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_job_manager() -> TrainingJobManager:
    return _JOB_MANAGER


def list_jobs_summary() -> list[dict[str, Any]]:
    _JOB_MANAGER.poll_updates()
    rows: list[dict[str, Any]] = []
    for job in _JOB_MANAGER.list_jobs():
        rows.append(
            {
                "job_id": job.job_id,
                "name": job.name,
                "status": str(job.status),
                "status_label": str(job.status_label),
                "duration": job.duration_str,
                "best_score": "-" if job.best_score is None else f"{job.best_score:.4f}",
                "target_metric": str(job.target_metric),
                "mlflow_run_id": job.mlflow_run_id or "",
                "error_msg": job.error_msg or "",
                "logs_tail": "\n".join(job.logs[-40:]) if job.logs else "",
                "trials_count": len(job.trials_data),
                "model_summaries": json.dumps(job.model_summaries, default=str) if job.model_summaries else "{}",
            }
        )
    return rows


def submit_classical_job(
    *,
    task: str,
    target: str,
    train_dataset: str,
    train_version: str,
    test_dataset: str | None,
    test_version: str | None,
    experiment_name: str,
    preset: str,
    n_trials: int | None,
    timeout: int | None,
    time_budget: int | None,
    selected_models: list[str] | None,
    optimization_mode: str,
    optimization_metric: str,
    validation_strategy: str,
    validation_params: dict[str, Any] | None,
    random_state: int,
    early_stopping: int,
    use_ensemble: bool,
    use_deep_learning: bool,
    ensemble_mode: str,
    manual_params: dict[str, Any] | None,
    ensemble_config: dict[str, Any] | None,
    ensemble_configs_list: list[dict[str, Any]] | None,
    model_source: str,
) -> str:
    lake = DataLake(PROJECT_ROOT / "data_lake")
    train_df = lake.load_version(train_dataset, train_version)
    test_df = None
    if test_dataset and test_version:
        test_df = lake.load_version(test_dataset, test_version)

    clean_name = experiment_name.strip() or f"Reflex_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_config: dict[str, Any] = {
        "task": task,
        "target": target,
        "date_col": None,
        "train_df": train_df,
        "test_df": test_df,
        "preset": preset,
        "n_trials": n_trials,
        "timeout": timeout,
        "time_budget": time_budget,
        "selected_models": selected_models or None,
        "manual_params": manual_params or {},
        "experiment_name": clean_name,
        "random_state": random_state,
        "validation_strategy": validation_strategy,
        "validation_params": validation_params or {},
        "ensemble_config": ensemble_config or {},
        "ensemble_configs_list": ensemble_configs_list or [],
        "model_source": model_source,
        "optimization_mode": optimization_mode,
        "optimization_metric": optimization_metric,
        "nlp_config": {},
        "selected_nlp_cols": [],
        "early_stopping": early_stopping,
        "forecast_horizon": 1,
        "target_metric_name": optimization_metric.upper(),
        "stability_config": None,
        "use_ensemble": use_ensemble,
        "use_deep_learning": use_deep_learning,
        "ensemble_mode": ensemble_mode,
    }
    return _JOB_MANAGER.submit_job(job_config, name=clean_name)


def pause_job(job_id: str):
    _JOB_MANAGER.pause_job(job_id)


def resume_job(job_id: str):
    _JOB_MANAGER.resume_job(job_id)


def cancel_job(job_id: str):
    _JOB_MANAGER.cancel_job(job_id)


def delete_job(job_id: str):
    _JOB_MANAGER.delete_job(job_id)


def run_data_drift(
    reference_dataset: str,
    reference_version: str,
    current_dataset: str,
    current_version: str,
    threshold: float = 0.05,
) -> list[dict[str, Any]]:
    lake = DataLake(PROJECT_ROOT / "data_lake")
    reference_df = lake.load_version(reference_dataset, reference_version)
    current_df = lake.load_version(current_dataset, current_version)
    drift = DriftDetector.detect_drift(reference_df, current_df, threshold=threshold)
    rows: list[dict[str, Any]] = []
    for feature, details in drift.items():
        rows.append(
            {
                "feature": feature,
                "p_value": float(details.get("p_value", 1.0)),
                "drift_detected": bool(details.get("drift_detected", False)),
            }
        )
    rows.sort(key=lambda item: item["p_value"])
    return rows


def run_stability_analysis(
    *,
    model_name: str,
    model_version: str | None,
    dataset: str,
    version: str,
    target_column: str,
    task_type: str,
    run_seed: bool,
    run_split: bool,
    run_noise: bool,
) -> dict[str, Any]:
    model = load_registered_model(model_name, model_version)
    if model is None:
        raise RuntimeError("Could not load model from registry.")

    lake = DataLake(PROJECT_ROOT / "data_lake")
    df = lake.load_version(dataset, version)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    processor = AutoMLDataProcessor(target_column=target_column, task_type=task_type)
    X, y = processor.fit_transform(df)
    analyzer = StabilityAnalyzer(base_model=model, X=X, y=y, task_type=task_type)

    report: dict[str, Any] = {}
    if run_seed:
        seed_df = analyzer.run_seed_stability(n_iterations=5)
        report["seed_stability"] = seed_df.to_dict(orient="records")
    if run_split:
        split_df = analyzer.run_split_stability(n_splits=5)
        report["split_stability"] = split_df.to_dict(orient="records")
    if run_noise:
        noise_df = analyzer.run_noise_injection_stability(noise_level=0.05, n_iterations=5)
        report["noise_stability"] = noise_df.to_dict(orient="records")
    return report


def train_cv_model(
    *,
    task_type: str,
    num_classes: int,
    backbone: str,
    data_dir: str,
    label_csv: str | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    trainer = CVAutoMLTrainer(task_type=task_type, num_classes=num_classes, backbone=backbone)
    trained = trainer.train(
        data_dir=data_dir,
        n_epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        label_csv=label_csv,
    )
    if trained is None:
        raise RuntimeError("CV training did not return a model.")
    return {
        "task_type": task_type,
        "num_classes": num_classes,
        "backbone": backbone,
        "epochs": epochs,
        "history": trainer.history,
        "class_names": trainer.class_names,
    }


def get_registry_snapshot() -> list[dict[str, Any]]:
    models = get_registered_models()
    rows: list[dict[str, Any]] = []
    for model in models:
        latest_version = "-"
        latest_stage = "Unassigned"
        latest_versions = getattr(model, "latest_versions", []) or []
        if latest_versions:
            latest_version = str(getattr(latest_versions[0], "version", "-"))
            latest_stage = str(getattr(latest_versions[0], "current_stage", "Unassigned") or "Unassigned")
        rows.append(
            {
                "name": str(getattr(model, "name", "")),
                "version": latest_version,
                "stage": latest_stage,
                "description": str(getattr(model, "description", "") or "No description"),
            }
        )
    return rows


def register_from_run(run_id: str, model_name: str) -> bool:
    return register_model_from_run(run_id, model_name)


def get_model_details_json(model_name: str, version: str | None = None) -> str:
    details = get_model_details(model_name, version if version else None)
    return json.dumps(details or {}, indent=2, default=str)


def get_run_details_json(run_id: str) -> str:
    return json.dumps(get_run_details(run_id), indent=2, default=str)


def deploy_model_to_hf(model_path: str, repo_id: str, token: str, private_repo: bool = False) -> str:
    return deploy_to_huggingface(
        model_path=model_path,
        repo_id=repo_id,
        token=token,
        private=private_repo,
        model_card_data={"source": "AutoMLOps Studio Reflex"},
    )


def send_api_inference(url: str, api_key: str, payload_json: str) -> str:
    payload = json.loads(payload_json)
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object with key 'data'.")
    response = requests.post(
        url,
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text}
    return json.dumps({"status_code": response.status_code, "response": body}, indent=2, default=str)


def list_available_models_for_task(task: str, use_ensemble: bool = True, use_deep_learning: bool = True) -> list[str]:
    trainer = AutoMLTrainer(task_type=task, use_ensemble=use_ensemble, use_deep_learning=use_deep_learning)
    return [model for model in trainer.get_available_models() if model not in NON_BASE_MODEL_KEYS]


def get_manual_param_cards(task: str, model_names: list[str]) -> list[dict[str, Any]]:
    trainer = AutoMLTrainer(task_type=task)
    fields: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw_name in model_names:
        model_name = str(raw_name).strip()
        if not model_name or model_name in seen or model_name in NON_BASE_MODEL_KEYS:
            continue
        seen.add(model_name)

        schema = trainer.get_model_params_schema(model_name)
        if not schema:
            continue

        schema_fields: list[dict[str, Any]] = []
        for param_name, spec in schema.items():
            if not spec:
                continue

            param_type = str(spec[0])
            field: dict[str, Any] = {
                "model": model_name,
                "name": param_name,
                "label": param_name.replace("_", " ").title(),
                "kind": "select" if param_type == "list" else param_type,
                "current_value": _stringify_schema_value(spec[-1] if len(spec) > 1 else ""),
            }

            if param_type == "list":
                options = spec[1] if len(spec) > 1 else []
                field["options"] = [_stringify_schema_value(option) for option in options]
                field["options_text"] = ", ".join(field["options"])
            elif param_type in {"int", "float"} and len(spec) >= 4:
                field["min_value"] = spec[1]
                field["max_value"] = spec[2]

            schema_fields.append(field)

        field_count = len(schema_fields)
        for index, field in enumerate(schema_fields):
            fields.append(
                {
                    **field,
                    "model_label": _humanize_model_name(model_name),
                    "field_count": field_count,
                    "is_first": index == 0,
                }
            )

    return fields