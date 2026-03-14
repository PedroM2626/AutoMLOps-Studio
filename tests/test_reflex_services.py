from pathlib import Path

import pandas as pd

from automlops_reflex import services
from src.core.data_lake import DataLake


class DummyVersion:
    def __init__(self, version: str, stage: str):
        self.version = version
        self.current_stage = stage


class DummyModel:
    def __init__(self, name: str):
        self.name = name
        self.description = "model description"
        self.latest_versions = [DummyVersion("3", "Production")]


def test_summarize_project_collects_catalog_and_tracking(monkeypatch, tmp_path: Path):
    lake = DataLake(tmp_path / "data_lake")
    lake.save_dataframe(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), "training_set")

    monitoring_dir = tmp_path / "data_lake" / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "feature": [1.0],
            "__timestamp": ["2026-03-14T10:00:00"],
        }
    ).to_csv(monitoring_dir / "api_telemetry.csv", index=False)

    monkeypatch.setattr(services, "get_registered_models", lambda: [DummyModel("fraud_detector")])
    monkeypatch.setattr(
        services,
        "get_all_runs",
        lambda: pd.DataFrame(
            [
                {
                    "run_id": "1234567890abcdef",
                    "experiment_name": "exp_a",
                    "status": "FINISHED",
                    "start_time": "2026-03-14T09:00:00",
                    "metrics.accuracy": 0.98,
                }
            ]
        ),
    )
    monkeypatch.setattr(services, "list_service_statuses", lambda: {name: {"status": "stopped", "status_label": "Stopped", "pid": "-", "url": spec["url"], "ready": False} for name, spec in services.SERVICE_SPECS.items()})

    summary = services.summarize_project(tmp_path)

    assert summary["dataset_count"] == 1
    assert summary["dataset_version_count"] == 1
    assert summary["registered_model_count"] == 1
    assert summary["run_count"] == 1
    assert summary["telemetry_row_count"] == 1
    assert summary["datasets"][0]["name"] == "training_set"
    assert summary["models"][0]["name"] == "fraud_detector"
    assert summary["runs"][0]["experiment"] == "exp_a"


def test_connect_dagshub_sets_tracking_uri(monkeypatch):
    state = {"uri": "sqlite:///mlflow.db"}

    monkeypatch.setattr(services.mlflow, "set_tracking_uri", lambda uri: state.update({"uri": uri}))
    monkeypatch.setattr(services.mlflow, "get_tracking_uri", lambda: state["uri"])
    monkeypatch.setattr(services.mlflow, "search_experiments", lambda max_results=1: [])

    result = services.connect_dagshub("alice", "automlops", "secret-token")

    assert result["is_dagshub"] is True
    assert "dagshub.com/alice/automlops.mlflow" in str(result["uri"])
    assert services.os.environ.get("MLFLOW_TRACKING_USERNAME") == "alice"
    assert services.os.environ.get("MLFLOW_TRACKING_PASSWORD") == "secret-token"


def test_disconnect_dagshub_restores_local(monkeypatch):
    state = {"uri": "https://dagshub.com/alice/automlops.mlflow"}

    monkeypatch.setattr(services.mlflow, "set_tracking_uri", lambda uri: state.update({"uri": uri}))
    monkeypatch.setattr(services.mlflow, "get_tracking_uri", lambda: state["uri"])

    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "alice")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret-token")

    result = services.disconnect_dagshub()

    assert result["is_dagshub"] is False
    assert result["uri"] == "sqlite:///mlflow.db"
    assert "MLFLOW_TRACKING_USERNAME" not in services.os.environ
    assert "MLFLOW_TRACKING_PASSWORD" not in services.os.environ


def test_model_catalog_hides_non_base_strategies(monkeypatch):
    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def get_available_models(self):
            return ["random_forest", "voting_ensemble", "custom_voting", "bagging", "xgboost"]

        def get_model_params_schema(self, model_name: str):
            if model_name == "random_forest":
                return {
                    "rf_n_estimators": ("int", 10, 500, 100),
                    "rf_bootstrap": ("list", [True, False], True),
                }
            return {}

    monkeypatch.setattr(services, "AutoMLTrainer", DummyTrainer)

    models = services.list_available_models_for_task("classification")
    cards = services.get_manual_param_cards("classification", ["random_forest", "custom_voting"])

    assert models == ["random_forest", "xgboost"]
    assert len(cards) == 2
    assert cards[0]["model"] == "random_forest"
    assert cards[0]["is_first"] is True
    assert cards[1]["options"] == ["True", "False"]