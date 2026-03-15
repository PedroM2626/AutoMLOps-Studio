from __future__ import annotations

import ast
from pathlib import Path
import json
import re
from typing import Any, TypedDict

import reflex as rx

from src.core.data_lake import DataLake
from .services import (
    list_service_statuses,
    start_service,
    stop_service,
    summarize_project,
    run_data_drift,
    submit_classical_job,
    list_jobs_summary,
    pause_job,
    resume_job,
    cancel_job,
    delete_job,
    get_registry_snapshot,
    get_model_details_json,
    register_from_run,
    get_run_details_json,
    deploy_model_to_hf,
    send_api_inference,
    run_stability_analysis,
    train_cv_model,
    list_available_models_for_task,
    get_manual_param_cards,
    get_tracking_status,
    connect_dagshub,
    disconnect_dagshub,
    NON_BASE_MODEL_KEYS,
)


class ManualParamField(TypedDict):
    name: str
    label: str
    kind: str
    value_text: str
    options: list[str]
    options_text: str
    min_value: str
    max_value: str


class ManualParamCard(TypedDict):
    model: str
    model_label: str
    field_count: str
    fields: list[ManualParamField]


class AppState(rx.State):
    active_module: str = "Overview"
    valid_modules: list[str] = [
        "Overview",
        "Data",
        "AutoML",
        "Experiments",
        "Registry & Deploy",
        "Monitoring",
        "Computer Vision",
    ]

    dataset_count: int = 0
    dataset_version_count: int = 0
    run_count: int = 0
    registered_model_count: int = 0
    telemetry_row_count: int = 0
    telemetry_last_seen: str = "No telemetry yet"
    generated_at: str = ""

    api_status: str = "stopped"
    api_status_label: str = "Stopped"
    api_pid: str = "-"
    api_url: str = "http://127.0.0.1:8000/docs"
    api_ready: bool = False

    mlflow_status: str = "stopped"
    mlflow_status_label: str = "Stopped"
    mlflow_pid: str = "-"
    mlflow_url: str = "http://127.0.0.1:5000"
    mlflow_ready: bool = False

    datasets: list[dict[str, str]] = []
    runs: list[dict[str, str]] = []
    registry_rows: list[dict[str, str]] = []
    jobs: list[dict[str, str]] = []
    consumption_code: str = ""

    dataset_name: str = ""
    selected_dataset: str = ""
    selected_version: str = ""
    dataset_preview_md: str = ""
    local_file_path: str = ""
    data_upload_status: str = ""
    data_split_strategy: str = "Random"
    data_train_percent: int = 80
    data_time_column: str = ""
    data_manual_split_column: str = ""
    data_schema_overrides_json: str = "[]"

    drift_reference_dataset: str = ""
    drift_reference_version: str = ""
    drift_current_dataset: str = ""
    drift_current_version: str = ""
    drift_results_json: str = "[]"

    automl_task: str = "classification"
    automl_target_column: str = ""
    automl_train_dataset: str = ""
    automl_train_version: str = ""
    automl_test_dataset: str = ""
    automl_test_version: str = ""
    automl_experiment_name: str = "Reflex_Experiment"

    automl_mode_selection: str = "Automatic (Preset)"
    automl_training_strategy: str = "Automatic"
    automl_ensemble_mode: str = "both"
    automl_use_deep_learning: bool = True
    automl_model_source: str = "Standard AutoML"
    automl_registry_model_name: str = ""
    automl_uploaded_model_path: str = ""
    automl_uploaded_model_status: str = ""

    automl_preset: str = "medium"
    automl_n_trials: int = 20
    automl_timeout: int = 300
    automl_time_budget: int = 1800
    automl_optimization_mode: str = "bayesian"
    automl_optimization_metric: str = "accuracy"

    automl_validation_strategy: str = "auto"
    automl_validation_folds: int = 5
    automl_validation_shuffle: bool = True
    automl_validation_test_size: int = 20
    automl_validation_gap: int = 0
    automl_validation_max_train_size: int = 0
    automl_validation_stratify: bool = True

    automl_random_state: int = 42
    automl_early_stopping: int = 10

    automl_models_csv: str = ""
    automl_selected_model_list: list[str] = []
    automl_manual_params_json: str = "{}"
    automl_available_models: list[str] = []
    automl_manual_param_cards: list[ManualParamCard] = []
    automl_submit_status: str = ""
    automl_custom_ensembles: list[dict[str, str]] = []
    automl_ensemble_counter: int = 0
    new_ensemble_type: str = "Voting"
    new_ensemble_models_csv: str = ""
    new_ensemble_meta_model: str = "logistic_regression"
    new_ensemble_voting_type: str = "soft"
    new_ensemble_bagging_base: str = "decision_tree"

    selected_job_id: str = ""
    selected_job_logs: str = ""
    selected_job_error: str = ""
    job_chart_rows: list[dict[str, str]] = []

    selected_model_name: str = ""
    selected_model_version: str = ""
    selected_run_id: str = ""
    model_details_json: str = "{}"
    run_details_json: str = "{}"
    register_model_name: str = ""
    register_result: str = ""

    hf_model_path: str = ""
    hf_repo_id: str = ""
    hf_token: str = ""
    hf_private: bool = False
    hf_result: str = ""

    api_predict_url: str = "http://127.0.0.1:8000/predict"
    api_key: str = "supersecretkey123"
    api_payload_json: str = '{"data": []}'
    api_response_json: str = ""

    stability_model_name: str = ""
    stability_model_version: str = ""
    stability_dataset: str = ""
    stability_version: str = ""
    stability_target: str = ""
    stability_task: str = "classification"
    stability_seed: bool = True
    stability_split: bool = True
    stability_noise: bool = False
    stability_result_json: str = "{}"

    cv_task_type: str = "image_classification"
    cv_num_classes: int = 2
    cv_backbone: str = "resnet18"
    cv_data_dir: str = ""
    cv_label_csv: str = ""
    cv_epochs: int = 2
    cv_batch_size: int = 8
    cv_learning_rate: float = 0.001
    cv_result_json: str = ""

    tracking_uri: str = ""
    dagshub_status_label: str = "Local MLflow"
    dagshub_connected: bool = False
    dagshub_username: str = ""
    dagshub_repo: str = ""
    dagshub_token: str = ""
    dagshub_message: str = ""

    dataset_versions_map: dict[str, list[str]] = {}
    dataset_columns_map: dict[str, list[str]] = {}
    model_versions_map: dict[str, list[str]] = {}
    cv_data_dir_options: list[str] = []
    cv_label_csv_map: dict[str, list[str]] = {}

    data_column_options: list[str] = []
    automl_target_options: list[str] = []
    stability_target_options: list[str] = []

    @rx.var
    def automl_metric_options(self) -> list[str]:
        mapping = {
            "classification": ["accuracy", "f1", "precision", "recall", "roc_auc"],
            "regression": ["r2", "rmse", "mae"],
            "clustering": ["silhouette"],
            "time_series": ["rmse", "mae", "mape"],
            "anomaly_detection": ["f1", "accuracy"],
        }
        return mapping.get(self.automl_task, ["accuracy"])

    @rx.var
    def dataset_options(self) -> list[str]:
        return list(self.dataset_versions_map.keys())

    @rx.var
    def dataset_options_optional(self) -> list[str]:
        return list(self.dataset_options)

    @rx.var
    def selected_dataset_version_options(self) -> list[str]:
        return self.dataset_versions_map.get(self.selected_dataset, [])

    @rx.var
    def data_column_options_optional(self) -> list[str]:
        return list(self.data_column_options)

    @rx.var
    def drift_reference_version_options(self) -> list[str]:
        return self.dataset_versions_map.get(self.drift_reference_dataset, [])

    @rx.var
    def drift_current_version_options(self) -> list[str]:
        return self.dataset_versions_map.get(self.drift_current_dataset, [])

    @rx.var
    def automl_train_version_options(self) -> list[str]:
        return self.dataset_versions_map.get(self.automl_train_dataset, [])

    @rx.var
    def automl_test_version_options(self) -> list[str]:
        return self.dataset_versions_map.get(self.automl_test_dataset, [])

    @rx.var
    def automl_test_version_options_optional(self) -> list[str]:
        return list(self.automl_test_version_options)

    @rx.var
    def stability_version_options(self) -> list[str]:
        return self.dataset_versions_map.get(self.stability_dataset, [])

    @rx.var
    def model_name_options(self) -> list[str]:
        names = [str(item.get("name", "")) for item in self.registry_rows if str(item.get("name", ""))]
        return sorted(list(dict.fromkeys(names)))

    @rx.var
    def selected_model_version_options(self) -> list[str]:
        return self.model_versions_map.get(self.selected_model_name, [])

    @rx.var
    def selected_model_version_options_optional(self) -> list[str]:
        return list(self.selected_model_version_options)

    @rx.var
    def stability_model_version_options(self) -> list[str]:
        return self.model_versions_map.get(self.stability_model_name, [])

    @rx.var
    def stability_model_version_options_optional(self) -> list[str]:
        return list(self.stability_model_version_options)

    @rx.var
    def register_model_name_options(self) -> list[str]:
        return self.model_name_options

    @rx.var
    def run_id_options(self) -> list[str]:
        run_ids = [str(item.get("run_id", "")) for item in self.runs if str(item.get("run_id", ""))]
        return list(dict.fromkeys(run_ids))

    @rx.var
    def job_id_options(self) -> list[str]:
        return [str(item.get("job_id", "")) for item in self.jobs if str(item.get("job_id", ""))]

    @rx.var
    def cv_label_csv_options(self) -> list[str]:
        return self.cv_label_csv_map.get(self.cv_data_dir, [])

    @rx.var
    def cv_data_dir_options_optional(self) -> list[str]:
        return list(self.cv_data_dir_options)

    @rx.var
    def cv_label_csv_options_optional(self) -> list[str]:
        return list(self.cv_label_csv_options)

    @rx.event
    def set_module(self, module_name: str):
        if module_name in self.valid_modules:
            self.active_module = module_name
        else:
            self.active_module = "Overview"

    @rx.event
    def initialize(self):
        for service_name in ("api", "mlflow"):
            try:
                start_service(service_name)
            except Exception:
                pass
        self.refresh()

    @rx.event
    def refresh(self):
        if self.active_module not in self.valid_modules:
            self.active_module = "Overview"

        snapshot = summarize_project()
        self.dataset_count = snapshot["dataset_count"]
        self.dataset_version_count = snapshot["dataset_version_count"]
        self.run_count = snapshot["run_count"]
        self.registered_model_count = snapshot["registered_model_count"]
        self.telemetry_row_count = snapshot["telemetry_row_count"]
        self.telemetry_last_seen = snapshot["telemetry_last_seen"]
        self.generated_at = snapshot["generated_at"]
        self.datasets = snapshot["datasets"]
        self.runs = snapshot["runs"]

        services = list_service_statuses()
        self._apply_service_statuses(services)

        try:
            self.registry_rows = get_registry_snapshot()
        except Exception as exc:
            self.registry_rows = []
            self.register_result = f"Failed to list registry: {exc}"

        try:
            self.jobs = list_jobs_summary()
        except Exception as exc:
            self.jobs = []
            self.automl_submit_status = f"Failed to list jobs: {exc}"

        self._refresh_dynamic_selectors()
        self._sync_job_selection()
        self._build_job_chart_rows()

        if not self.automl_available_models:
            self.refresh_models_for_task()

        self._refresh_tracking_status()

    @rx.event
    def start_service(self, service_name: str):
        start_service(service_name)
        self.refresh()

    @rx.event
    def stop_service(self, service_name: str):
        stop_service(service_name)
        self.refresh()

    @rx.event
    def refresh_models_for_task(self):
        try:
            models = list_available_models_for_task(
                self.automl_task,
                use_ensemble=(self.automl_ensemble_mode != "single"),
                use_deep_learning=self.automl_use_deep_learning,
            )
            self.automl_available_models = models[:60]
            self.automl_selected_model_list = [model for model in self.automl_selected_model_list if model in self.automl_available_models]
            self.automl_models_csv = ", ".join(self.automl_selected_model_list)
            self.refresh_manual_param_cards()
            if len(models) > 60:
                self.automl_submit_status = f"Showing 60 of {len(models)} available models to keep the interface responsive."
            else:
                self.automl_submit_status = f"{len(models)} models loaded."
        except Exception as exc:
            self.automl_available_models = []
            self.automl_submit_status = f"Error loading models: {exc}"

    @rx.event
    def update_automl_task(self, task: str):
        self.automl_task = task
        options = self.automl_metric_options
        if options:
            self.automl_optimization_metric = options[0]
        if task == "time_series":
            self.automl_validation_strategy = "time_series_cv"
            self.automl_validation_shuffle = False
        self.refresh_models_for_task()

    @rx.event
    def update_selected_dataset(self, dataset_name: str):
        self.selected_dataset = dataset_name
        versions = self.dataset_versions_map.get(dataset_name, [])
        self.selected_version = versions[0] if versions else ""
        self._refresh_data_columns()

    @rx.event
    def update_selected_version(self, version: str):
        self.selected_version = version
        self._refresh_data_columns()

    @rx.event
    def update_drift_reference_dataset(self, dataset_name: str):
        self.drift_reference_dataset = dataset_name
        versions = self.dataset_versions_map.get(dataset_name, [])
        self.drift_reference_version = versions[0] if versions else ""

    @rx.event
    def update_drift_current_dataset(self, dataset_name: str):
        self.drift_current_dataset = dataset_name
        versions = self.dataset_versions_map.get(dataset_name, [])
        self.drift_current_version = versions[0] if versions else ""

    @rx.event
    def update_automl_train_dataset(self, dataset_name: str):
        self.automl_train_dataset = dataset_name
        versions = self.dataset_versions_map.get(dataset_name, [])
        self.automl_train_version = versions[0] if versions else ""
        self._refresh_automl_target_options()

    @rx.event
    def update_automl_train_version(self, version: str):
        self.automl_train_version = version
        self._refresh_automl_target_options()

    @rx.event
    def update_automl_test_dataset(self, dataset_name: str):
        self.automl_test_dataset = dataset_name
        versions = self.dataset_versions_map.get(dataset_name, [])
        self.automl_test_version = versions[0] if versions else ""

    @rx.event
    def update_selected_job(self, job_id: str):
        self.selected_job_id = job_id
        self._sync_job_selection()

    @rx.event
    def update_selected_model_name(self, model_name: str):
        self.selected_model_name = model_name
        versions = self.model_versions_map.get(model_name, [])
        self.selected_model_version = versions[0] if versions else ""

    @rx.event
    def update_stability_model_name(self, model_name: str):
        self.stability_model_name = model_name
        versions = self.model_versions_map.get(model_name, [])
        self.stability_model_version = versions[0] if versions else ""

    @rx.event
    def update_stability_dataset(self, dataset_name: str):
        self.stability_dataset = dataset_name
        versions = self.dataset_versions_map.get(dataset_name, [])
        self.stability_version = versions[0] if versions else ""
        self._refresh_stability_target_options()

    @rx.event
    def update_stability_version(self, version: str):
        self.stability_version = version
        self._refresh_stability_target_options()

    @rx.event
    def update_cv_data_dir(self, data_dir: str):
        self.cv_data_dir = data_dir
        csv_options = self.cv_label_csv_map.get(data_dir, [])
        if csv_options and self.cv_label_csv not in csv_options:
            self.cv_label_csv = csv_options[0]

    @rx.event
    def update_mode_selection(self, mode: str):
        self.automl_mode_selection = mode
        if mode == "Automatic (Preset)":
            self.automl_models_csv = ""
            self.automl_selected_model_list = []
            self.automl_manual_param_cards = []
        else:
            self.refresh_manual_param_cards()

    @rx.event
    def update_training_strategy(self, strategy: str):
        self.automl_training_strategy = strategy
        if strategy == "Manual":
            self.refresh_manual_param_cards()
        else:
            self.automl_manual_param_cards = []

    @rx.event
    def update_model_source(self, source: str):
        self.automl_model_source = source
        if source != "Standard AutoML":
            self.automl_manual_param_cards = []
        else:
            self.refresh_manual_param_cards()

    @rx.event
    def update_validation_strategy(self, strategy: str):
        self.automl_validation_strategy = strategy
        if strategy == "time_series_cv":
            self.automl_validation_shuffle = False

    @rx.event
    def set_automl_models_csv_input(self, value: str):
        models = self._normalize_model_list(value)
        self.automl_selected_model_list = models
        self.automl_models_csv = ", ".join(models)
        self.refresh_manual_param_cards()

    @rx.event
    def toggle_automl_model(self, model_name: str):
        models = list(self.automl_selected_model_list)
        if model_name in models:
            models = [item for item in models if item != model_name]
        else:
            models.append(model_name)
        self.automl_selected_model_list = models
        self.automl_models_csv = ", ".join(models)
        self.refresh_manual_param_cards()

    @rx.event
    def update_ensemble_mode(self, mode: str):
        self.automl_ensemble_mode = mode
        self.refresh_models_for_task()

    @rx.event
    def update_use_deep_learning(self, value: bool):
        self.automl_use_deep_learning = value
        self.refresh_models_for_task()

    @rx.event
    def refresh_manual_param_cards(self):
        if self.automl_training_strategy != "Manual" or self.automl_mode_selection != "Manual (Select)" or self.automl_model_source != "Standard AutoML":
            self.automl_manual_param_cards = []
            return

        rows = get_manual_param_cards(self.automl_task, self.automl_selected_model_list)
        grouped: dict[str, ManualParamCard] = {}
        merged_params: dict[str, Any] = {}
        try:
            parsed_existing = json.loads(self.automl_manual_params_json or "{}")
            if isinstance(parsed_existing, dict):
                merged_params.update(parsed_existing)
        except Exception:
            pass

        for row in rows:
            model_name = str(row.get("model", ""))
            model_label = str(row.get("model_label", model_name))
            if model_name not in grouped:
                grouped[model_name] = {
                    "model": model_name,
                    "model_label": model_label,
                    "field_count": str(row.get("field_count", "0")),
                    "fields": [],
                }

            param_name = str(row.get("name", ""))
            kind = str(row.get("kind", "text"))
            default_value = self._parse_manual_param_value(
                str(row.get("current_value", "")),
                kind,
            )
            current_value = merged_params.get(param_name, default_value)
            merged_params[param_name] = current_value

            grouped[model_name]["fields"].append(
                {
                    "name": param_name,
                    "label": str(row.get("label", param_name.replace("_", " ").title())),
                    "kind": kind,
                    "value_text": self._stringify_manual_param_value(current_value),
                    "options": [str(option) for option in row.get("options", [])] if isinstance(row.get("options", []), list) else [],
                    "options_text": ", ".join([str(option) for option in row.get("options", [])]) if isinstance(row.get("options", []), list) else "",
                    "min_value": "" if row.get("min_value") in (None, "") else str(row.get("min_value")),
                    "max_value": "" if row.get("max_value") in (None, "") else str(row.get("max_value")),
                }
            )

        self.automl_manual_param_cards = list(grouped.values())
        self.automl_manual_params_json = json.dumps(merged_params, indent=2, default=str)

    @rx.event
    def update_manual_param_value(self, model_name: str, param_name: str, value: str):
        parsed: dict[str, Any] = {}
        try:
            loaded = json.loads(self.automl_manual_params_json or "{}")
            if isinstance(loaded, dict):
                parsed = loaded
        except Exception:
            parsed = {}

        param_kind = "text"
        for block in self.automl_manual_param_cards:
            if str(block.get("model", "")) != model_name:
                continue
            for field in block.get("fields", []):
                if str(field.get("name", "")) == param_name:
                    param_kind = str(field.get("kind", "text"))
                    break

        try:
            parsed[param_name] = self._parse_manual_param_value(value, param_kind)
        except Exception:
            parsed[param_name] = value

        for block in self.automl_manual_param_cards:
            if str(block.get("model", "")) != model_name:
                continue
            for field in block.get("fields", []):
                if str(field.get("name", "")) == param_name:
                    field["value_text"] = str(value)
                    break

        self.automl_manual_params_json = json.dumps(parsed, indent=2, default=str)

    @rx.event
    def add_custom_ensemble(self):
        models = [item.strip() for item in self.new_ensemble_models_csv.split(",") if item.strip()]
        if self.new_ensemble_type in ["Voting", "Stacking"] and len(models) < 2:
            self.automl_submit_status = "Custom ensemble needs at least 2 base models."
            return

        payload = {
            "id": str(self.automl_ensemble_counter),
            "type": self.new_ensemble_type.lower(),
            "models": ",".join(models),
            "meta_model": self.new_ensemble_meta_model,
            "voting_type": self.new_ensemble_voting_type,
            "bagging_base": self.new_ensemble_bagging_base,
        }
        self.automl_custom_ensembles = [*self.automl_custom_ensembles, payload]
        self.automl_ensemble_counter += 1
        self.new_ensemble_models_csv = ""

    @rx.event
    def remove_custom_ensemble(self, ensemble_id: str):
        self.automl_custom_ensembles = [item for item in self.automl_custom_ensembles if str(item.get("id", "")) != str(ensemble_id)]

    @rx.event
    async def handle_model_upload(self, files: list[rx.UploadFile]):
        model_dir = Path("./models/uploaded")
        model_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            try:
                data = await file.read()
                fname = file.filename or "uploaded_model.pkl"
                out_path = model_dir / fname
                out_path.write_bytes(data)
                self.automl_uploaded_model_path = str(out_path)
                self.automl_uploaded_model_status = f"Model attached: {fname}"
                return
            except Exception as exc:
                self.automl_uploaded_model_status = f"Model upload failed: {exc}"
                return

    @rx.event
    async def handle_data_upload(self, files: list[rx.UploadFile]):
        saved: list[str] = []
        failed: list[str] = []
        lake = DataLake("./data_lake")
        for file in files:
            try:
                payload = await file.read()
                filename = file.filename or "uploaded_file"
                dataset_name = Path(filename).stem
                dataset_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", dataset_name) or "dataset"
                lake.save_raw_file(payload, dataset_name, filename)
                saved.append(filename)
            except Exception:
                failed.append(file.filename or "unknown")

        parts: list[str] = []
        if saved:
            parts.append(f"{len(saved)} file(s) saved: {', '.join(saved[:5])}")
        if failed:
            parts.append(f"Failed on {len(failed)} file(s): {', '.join(failed[:5])}")
        self.data_upload_status = " | ".join(parts) if parts else "No files uploaded."
        self.refresh()

    @rx.event
    def save_local_file_to_lake(self):
        path = Path(self.local_file_path.strip())
        if not path.exists() or not path.is_file():
            self.dataset_preview_md = "Local file not found."
            return

        dataset_name = self.dataset_name.strip() or path.stem
        payload = path.read_bytes()
        try:
            lake = DataLake("./data_lake")
            lake.save_raw_file(payload, dataset_name, path.name)
            self.dataset_preview_md = f"File saved to Data Lake: {dataset_name}"
            self.refresh()
        except Exception as exc:
            self.dataset_preview_md = f"Error saving to Data Lake: {exc}"

    @rx.event
    def preview_dataset(self):
        if not self.selected_dataset or not self.selected_version:
            self.dataset_preview_md = "Select dataset and version."
            return

        try:
            lake = DataLake("./data_lake")
            df = lake.load_version(self.selected_dataset, self.selected_version, nrows=20)
            self.dataset_preview_md = df.to_markdown(index=False)
        except Exception as exc:
            self.dataset_preview_md = f"Error in preview: {exc}"

    @rx.event
    def delete_selected_version(self):
        if not self.selected_dataset or not self.selected_version:
            return
        try:
            lake = DataLake("./data_lake")
            lake.delete_version(self.selected_dataset, self.selected_version)
            self.selected_version = ""
            self.dataset_preview_md = "Version removed."
            self.refresh()
        except Exception as exc:
            self.dataset_preview_md = f"Error removing version: {exc}"

    @rx.event
    def run_drift_detection(self):
        if not (self.drift_reference_dataset and self.drift_reference_version and self.drift_current_dataset and self.drift_current_version):
            self.drift_results_json = json.dumps({"error": "Select reference and current datasets."}, indent=2)
            return

        try:
            results = run_data_drift(
                reference_dataset=self.drift_reference_dataset,
                reference_version=self.drift_reference_version,
                current_dataset=self.drift_current_dataset,
                current_version=self.drift_current_version,
            )
            self.drift_results_json = json.dumps(results, indent=2, default=str)
        except Exception as exc:
            self.drift_results_json = json.dumps({"error": str(exc)}, indent=2)

    @rx.event
    def submit_automl(self):
        models = list(self.automl_selected_model_list) if self.automl_selected_model_list else self._normalize_model_list(self.automl_models_csv)
        selected_models = None if self.automl_mode_selection == "Automatic (Preset)" else (models if models else None)
        validation_params = {
            "folds": self.automl_validation_folds,
            "shuffle": self.automl_validation_shuffle,
            "test_size": self.automl_validation_test_size / 100.0,
            "gap": self.automl_validation_gap,
            "max_train_size": self.automl_validation_max_train_size or None,
            "stratify": self.automl_validation_stratify,
            "split_strategy": self.data_split_strategy,
            "train_percent": self.data_train_percent,
            "time_column": self.data_time_column,
            "manual_split_column": self.data_manual_split_column,
        }

        manual_params = {}
        if self.automl_training_strategy == "Manual":
            try:
                parsed = json.loads(self.automl_manual_params_json or "{}")
                if isinstance(parsed, dict):
                    manual_params.update(parsed)
            except Exception:
                self.automl_submit_status = "Invalid manual params JSON."
                return

        ensemble_config: dict[str, str | list[str] | None] = {}
        ensemble_configs_list: list[dict[str, str | list[str] | None]] = []
        if self.automl_custom_ensembles and self.automl_ensemble_mode != "single":
            for ens in self.automl_custom_ensembles:
                ens_type = ens.get("type", "")
                ens_models = [m.strip() for m in str(ens.get("models", "")).split(",") if m.strip()]
                if ens_type == "voting":
                    if selected_models is None:
                        selected_models = []
                    selected_models.append("custom_voting")
                    cfg = {
                        "voting_estimators": ens_models,
                        "voting_type": ens.get("voting_type", "soft"),
                        "voting_weights": None,
                    }
                    ensemble_configs_list.append(cfg)
                elif ens_type == "stacking":
                    if selected_models is None:
                        selected_models = []
                    selected_models.append("custom_stacking")
                    cfg = {
                        "stacking_estimators": ens_models,
                        "stacking_final_estimator": ens.get("meta_model", "logistic_regression"),
                    }
                    ensemble_configs_list.append(cfg)
                elif ens_type == "bagging":
                    if selected_models is None:
                        selected_models = []
                    selected_models.append("custom_bagging")
                    cfg = {
                        "bagging_base_estimator": ens.get("bagging_base", "decision_tree"),
                    }
                    ensemble_configs_list.append(cfg)

            if ensemble_configs_list:
                ensemble_config = ensemble_configs_list[0]

        if self.automl_model_source == "Model Registry":
            if self.automl_registry_model_name.strip():
                selected_models = [self.automl_registry_model_name.strip()]
            else:
                self.automl_submit_status = "Select a model from registry."
                return

        if self.automl_model_source == "Local Upload":
            if not self.automl_uploaded_model_path.strip():
                self.automl_submit_status = "Attach a local model to continue."
                return
            manual_params["uploaded_model_path"] = self.automl_uploaded_model_path

        try:
            job_id = submit_classical_job(
                task=self.automl_task,
                target=self.automl_target_column,
                train_dataset=self.automl_train_dataset,
                train_version=self.automl_train_version,
                test_dataset=self.automl_test_dataset or None,
                test_version=self.automl_test_version or None,
                experiment_name=self.automl_experiment_name,
                preset=self.automl_preset,
                n_trials=self.automl_n_trials,
                timeout=self.automl_timeout,
                time_budget=self.automl_time_budget,
                selected_models=selected_models,
                optimization_mode=self.automl_optimization_mode,
                optimization_metric=self.automl_optimization_metric,
                validation_strategy=self.automl_validation_strategy,
                validation_params=validation_params,
                random_state=self.automl_random_state,
                early_stopping=self.automl_early_stopping,
                use_ensemble=(self.automl_ensemble_mode != "single"),
                use_deep_learning=self.automl_use_deep_learning,
                ensemble_mode=self.automl_ensemble_mode,
                manual_params=manual_params,
                ensemble_config=ensemble_config,
                ensemble_configs_list=ensemble_configs_list,
                model_source=self.automl_model_source,
            )
            self.automl_submit_status = f"Job submitted: {job_id}"
            self.refresh()
        except Exception as exc:
            self.automl_submit_status = f"Error submitting job: {exc}"

    @rx.event
    def refresh_jobs(self):
        try:
            self.jobs = list_jobs_summary()
            self._sync_job_selection()
            self._build_job_chart_rows()
        except Exception as exc:
            self.automl_submit_status = f"Error updating jobs: {exc}"

    @rx.event
    def pause_selected_job(self):
        if self.selected_job_id:
            try:
                pause_job(self.selected_job_id)
                self.refresh_jobs()
            except Exception as exc:
                self.selected_job_error = str(exc)

    @rx.event
    def resume_selected_job(self):
        if self.selected_job_id:
            try:
                resume_job(self.selected_job_id)
                self.refresh_jobs()
            except Exception as exc:
                self.selected_job_error = str(exc)

    @rx.event
    def cancel_selected_job(self):
        if self.selected_job_id:
            try:
                cancel_job(self.selected_job_id)
                self.refresh_jobs()
            except Exception as exc:
                self.selected_job_error = str(exc)

    @rx.event
    def delete_selected_job(self):
        if self.selected_job_id:
            try:
                delete_job(self.selected_job_id)
                self.selected_job_id = ""
                self.selected_job_logs = ""
                self.selected_job_error = ""
                self.refresh_jobs()
            except Exception as exc:
                self.selected_job_error = str(exc)

    @rx.event
    def load_model_details(self):
        if self.selected_model_name:
            try:
                version = self.selected_model_version.strip() or None
                self.model_details_json = get_model_details_json(self.selected_model_name, version)
            except Exception as exc:
                self.model_details_json = json.dumps({"error": str(exc)}, indent=2)

    @rx.event
    def load_run_details(self):
        if self.selected_run_id:
            try:
                self.run_details_json = get_run_details_json(self.selected_run_id)
            except Exception as exc:
                self.run_details_json = json.dumps({"error": str(exc)}, indent=2)

    @rx.event
    def register_selected_run(self):
        if not self.selected_run_id or not self.register_model_name.strip():
            self.register_result = "Provide run_id and model name."
            return
        try:
            ok = register_from_run(self.selected_run_id, self.register_model_name.strip())
            self.register_result = "Model registered successfully." if ok else "Failed to register model."
            self.refresh()
        except Exception as exc:
            self.register_result = f"Error in registration: {exc}"

    @rx.event
    def deploy_to_hf(self):
        try:
            self.hf_result = deploy_model_to_hf(
                model_path=self.hf_model_path,
                repo_id=self.hf_repo_id,
                token=self.hf_token,
                private_repo=self.hf_private,
            )
        except Exception as exc:
            self.hf_result = f"Error in deploy: {exc}"

    @rx.event
    def send_predict_request(self):
        try:
            self.api_response_json = send_api_inference(
                url=self.api_predict_url,
                api_key=self.api_key,
                payload_json=self.api_payload_json,
            )
        except Exception as exc:
            self.api_response_json = json.dumps({"error": str(exc)}, indent=2)

    @rx.event
    def run_stability(self):
        try:
            report = run_stability_analysis(
                model_name=self.stability_model_name,
                model_version=self.stability_model_version or None,
                dataset=self.stability_dataset,
                version=self.stability_version,
                target_column=self.stability_target,
                task_type=self.stability_task,
                run_seed=self.stability_seed,
                run_split=self.stability_split,
                run_noise=self.stability_noise,
            )
            self.stability_result_json = json.dumps(report, indent=2, default=str)
        except Exception as exc:
            self.stability_result_json = json.dumps({"error": str(exc)}, indent=2)

    @rx.event
    def run_cv_training(self):
        try:
            report = train_cv_model(
                task_type=self.cv_task_type,
                num_classes=self.cv_num_classes,
                backbone=self.cv_backbone,
                data_dir=self.cv_data_dir,
                label_csv=self.cv_label_csv or None,
                epochs=self.cv_epochs,
                batch_size=self.cv_batch_size,
                learning_rate=self.cv_learning_rate,
            )
            self.cv_result_json = json.dumps(report, indent=2, default=str)
        except Exception as exc:
            self.cv_result_json = json.dumps({"error": str(exc)}, indent=2)

    @rx.event
    def load_consumption_code(self):
        """Generates a snippet for model consumption."""
        model_name = self.selected_model_name or "model"
        self.consumption_code = f"""import joblib
import pandas as pd

# 1. Load the pipeline
pipeline = joblib.load("{model_name}.pkl")

# 2. Prepare data
data = pd.DataFrame([{{
    "feature1": 1.0,
    "feature2": "Value"
}}])

# 3. Predict
predictions = pipeline.predict(data)
print(f"Predictions: {{predictions}}")
"""

    @rx.event
    def connect_dagshub_tracking(self):
        try:
            result = connect_dagshub(self.dagshub_username, self.dagshub_repo, self.dagshub_token)
            self._apply_tracking_status(result)
            self.dagshub_message = "Successfully connected to DagsHub."
            self.refresh()
        except Exception as exc:
            self.dagshub_message = f"Failed to connect to DagsHub: {exc}"

    @rx.event
    def disconnect_dagshub_tracking(self):
        try:
            result = disconnect_dagshub()
            self._apply_tracking_status(result)
            self.dagshub_message = "DagsHub connection removed."
            self.refresh()
        except Exception as exc:
            self.dagshub_message = f"Failed to disconnect DagsHub: {exc}"

    @rx.event
    def set_automl_n_trials_input(self, value: str):
        try:
            self.automl_n_trials = int(float(value))
        except Exception:
            pass

    @rx.event
    def set_automl_timeout_input(self, value: str):
        try:
            self.automl_timeout = int(float(value))
        except Exception:
            pass

    @rx.event
    def set_automl_time_budget_input(self, value: str):
        try:
            self.automl_time_budget = int(float(value))
        except Exception:
            pass

    @rx.event
    def set_automl_validation_folds_input(self, value: str):
        try:
            self.automl_validation_folds = int(float(value))
        except Exception:
            pass

    @rx.event
    def set_automl_validation_test_size_input(self, value: str):
        try:
            parsed = int(float(value))
            if 5 <= parsed <= 50:
                self.automl_validation_test_size = parsed
        except Exception:
            pass

    @rx.event
    def set_automl_validation_gap_input(self, value: str):
        try:
            self.automl_validation_gap = max(0, int(float(value)))
        except Exception:
            pass

    @rx.event
    def set_automl_validation_max_train_size_input(self, value: str):
        try:
            self.automl_validation_max_train_size = max(0, int(float(value)))
        except Exception:
            pass

    @rx.event
    def set_automl_random_state_input(self, value: str):
        try:
            self.automl_random_state = int(float(value))
        except Exception:
            pass

    @rx.event
    def set_automl_early_stopping_input(self, value: str):
        try:
            self.automl_early_stopping = int(float(value))
        except Exception:
            pass

    @rx.event
    def set_cv_num_classes_input(self, value: str):
        try:
            self.cv_num_classes = int(float(value))
        except Exception:
            pass

    @rx.event
    def set_cv_epochs_input(self, value: str):
        try:
            self.cv_epochs = int(float(value))
        except Exception:
            pass

    @rx.event
    def set_cv_batch_size_input(self, value: str):
        try:
            self.cv_batch_size = int(float(value))
        except Exception:
            pass

    @rx.event
    def set_cv_learning_rate_input(self, value: str):
        try:
            self.cv_learning_rate = float(value)
        except Exception:
            pass

    def _apply_service_statuses(self, statuses: dict[str, dict[str, str | bool]]):
        api = statuses.get("api", {})
        self.api_status = str(api.get("status", "stopped"))
        self.api_status_label = str(api.get("status_label", "Stopped"))
        self.api_pid = str(api.get("pid", "-"))
        self.api_url = str(api.get("url", "http://127.0.0.1:8000/docs"))
        self.api_ready = bool(api.get("ready", False))

        mlflow = statuses.get("mlflow", {})
        self.mlflow_status = str(mlflow.get("status", "stopped"))
        self.mlflow_status_label = str(mlflow.get("status_label", "Stopped"))
        self.mlflow_pid = str(mlflow.get("pid", "-"))
        self.mlflow_url = str(mlflow.get("url", "http://127.0.0.1:5000"))
        self.mlflow_ready = bool(mlflow.get("ready", False))

    def _sync_job_selection(self):
        if not self.jobs:
            self.selected_job_logs = ""
            self.selected_job_error = ""
            self.job_chart_rows = []
            return

        if not self.selected_job_id:
            self.selected_job_id = self.jobs[0].get("job_id", "")

        selected = next((item for item in self.jobs if item.get("job_id") == self.selected_job_id), None)
        if selected is None:
            selected = self.jobs[0]
            self.selected_job_id = selected.get("job_id", "")

        self.selected_job_logs = selected.get("logs_tail", "")
        self.selected_job_error = selected.get("error_msg", "")

    def _build_job_chart_rows(self):
        rows: list[dict[str, str]] = []
        best = 0.0
        parsed: list[tuple[str, float]] = []
        for item in self.jobs:
            score_raw = str(item.get("best_score", "-")).strip()
            try:
                score = float(score_raw)
                parsed.append((str(item.get("name", item.get("job_id", "job"))), score))
                best = max(best, score)
            except Exception:
                continue
        if best <= 0:
            self.job_chart_rows = []
            return
        for name, score in parsed[:10]:
            pct = max(5, int((score / best) * 100))
            rows.append({"name": name[:36], "score": f"{score:.4f}", "width": f"{pct}%"})
        self.job_chart_rows = rows

    def _refresh_dynamic_selectors(self):
        self._refresh_dataset_metadata()
        self._refresh_model_versions_map()
        self._refresh_cv_path_options()
        self._sync_field_values_with_options()

    def _refresh_dataset_metadata(self):
        lake = DataLake("./data_lake")
        versions_map: dict[str, list[str]] = {}
        columns_map: dict[str, list[str]] = {}
        for dataset_name in lake.list_datasets():
            versions = lake.list_versions(dataset_name)
            versions_map[dataset_name] = versions
            if not versions:
                continue
            try:
                frame = lake.load_version(dataset_name, versions[0], nrows=5)
                columns_map[dataset_name] = [str(col) for col in frame.columns]
            except Exception:
                columns_map[dataset_name] = []
        self.dataset_versions_map = versions_map
        self.dataset_columns_map = columns_map

    def _refresh_model_versions_map(self):
        versions_map: dict[str, list[str]] = {}
        for row in self.registry_rows:
            name = str(row.get("name", "")).strip()
            version = str(row.get("version", "")).strip()
            if not name:
                continue
            if name not in versions_map:
                versions_map[name] = []
            if version and version != "-" and version not in versions_map[name]:
                versions_map[name].append(version)
        self.model_versions_map = versions_map

    def _refresh_cv_path_options(self):
        root = Path(".").resolve()
        base_dirs = [root / "data_lake", root / "uploaded_files"]
        discovered_dirs: list[str] = []
        csv_map: dict[str, list[str]] = {}
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

        for base in base_dirs:
            if not base.exists() or not base.is_dir():
                continue
            try:
                for candidate in list(base.rglob("*")):
                    if not candidate.is_dir():
                        continue
                    try:
                        files = [item for item in candidate.iterdir() if item.is_file()]
                    except Exception:
                        continue
                    has_images = any(file.suffix.lower() in image_exts for file in files)
                    has_class_subdirs = any(item.is_dir() for item in candidate.iterdir())
                    csv_files = sorted([str(file) for file in files if file.suffix.lower() == ".csv"])
                    if has_images or has_class_subdirs or csv_files:
                        candidate_str = str(candidate)
                        discovered_dirs.append(candidate_str)
                        if csv_files:
                            csv_map[candidate_str] = csv_files
            except Exception:
                continue

        unique_dirs = sorted(list(dict.fromkeys(discovered_dirs)))
        self.cv_data_dir_options = unique_dirs
        self.cv_label_csv_map = csv_map

    def _sync_field_values_with_options(self):
        self.selected_dataset = self._ensure_option(self.selected_dataset, self.dataset_options)
        self.selected_version = self._ensure_option(self.selected_version, self.selected_dataset_version_options)

        self.drift_reference_dataset = self._ensure_option(self.drift_reference_dataset, self.dataset_options)
        self.drift_reference_version = self._ensure_option(self.drift_reference_version, self.drift_reference_version_options)

        self.drift_current_dataset = self._ensure_option(self.drift_current_dataset, self.dataset_options)
        self.drift_current_version = self._ensure_option(self.drift_current_version, self.drift_current_version_options)

        self.automl_train_dataset = self._ensure_option(self.automl_train_dataset, self.dataset_options)
        self.automl_train_version = self._ensure_option(self.automl_train_version, self.automl_train_version_options)

        self.automl_test_dataset = self._ensure_option(self.automl_test_dataset, ["", *self.dataset_options])
        self.automl_test_version = self._ensure_option(self.automl_test_version, ["", *self.automl_test_version_options])

        self.selected_model_name = self._ensure_option(self.selected_model_name, self.model_name_options)
        self.selected_model_version = self._ensure_option(self.selected_model_version, ["", *self.selected_model_version_options])
        self.automl_registry_model_name = self._ensure_option(self.automl_registry_model_name, self.model_name_options)

        self.stability_model_name = self._ensure_option(self.stability_model_name, self.model_name_options)
        self.stability_model_version = self._ensure_option(self.stability_model_version, ["", *self.stability_model_version_options])

        self.stability_dataset = self._ensure_option(self.stability_dataset, self.dataset_options)
        self.stability_version = self._ensure_option(self.stability_version, self.stability_version_options)

        self.selected_run_id = self._ensure_option(self.selected_run_id, self.run_id_options)
        self.selected_job_id = self._ensure_option(self.selected_job_id, self.job_id_options)

        self.cv_data_dir = self._ensure_option(self.cv_data_dir, ["", *self.cv_data_dir_options])
        self.cv_label_csv = self._ensure_option(self.cv_label_csv, ["", *self.cv_label_csv_options])

        self._refresh_data_columns()
        self._refresh_automl_target_options()
        self._refresh_stability_target_options()

    def _refresh_data_columns(self):
        columns = self.dataset_columns_map.get(self.selected_dataset, [])
        self.data_column_options = columns
        if self.data_time_column and self.data_time_column not in columns:
            self.data_time_column = ""
        if self.data_manual_split_column and self.data_manual_split_column not in columns:
            self.data_manual_split_column = ""

    def _refresh_automl_target_options(self):
        columns = self.dataset_columns_map.get(self.automl_train_dataset, [])
        self.automl_target_options = columns
        if not self.automl_target_column and columns:
            self.automl_target_column = columns[-1]
        elif self.automl_target_column and self.automl_target_column not in columns:
            self.automl_target_column = columns[-1] if columns else ""

    def _refresh_stability_target_options(self):
        columns = self.dataset_columns_map.get(self.stability_dataset, [])
        self.stability_target_options = columns
        if not self.stability_target and columns:
            self.stability_target = columns[-1]
        elif self.stability_target and self.stability_target not in columns:
            self.stability_target = columns[-1] if columns else ""

    def _ensure_option(self, current: str, options: list[str]) -> str:
        if current in options:
            return current
        return options[0] if options else ""

    def _refresh_tracking_status(self):
        try:
            status = get_tracking_status()
            self._apply_tracking_status(status)
        except Exception:
            self.tracking_uri = "unknown"
            self.dagshub_connected = False
            self.dagshub_status_label = "Tracking unavailable"

    def _apply_tracking_status(self, status: dict[str, str | bool]):
        self.tracking_uri = str(status.get("uri", ""))
        self.dagshub_connected = bool(status.get("is_dagshub", False))
        self.dagshub_status_label = str(status.get("status_label", "Local MLflow"))

    def _normalize_model_list(self, raw_value: str) -> list[str]:
        normalized: list[str] = []
        for item in raw_value.split(","):
            model_name = item.strip()
            if not model_name or model_name in NON_BASE_MODEL_KEYS or model_name in normalized:
                continue
            normalized.append(model_name)
        return normalized

    def _parse_manual_param_value(self, raw_value: str, kind: str):
        text = raw_value.strip()
        if kind == "select":
            return self._coerce_literal(text)
        if kind == "int":
            return int(float(text))
        if kind == "float":
            return float(text)
        return self._coerce_literal(text)

    def _coerce_literal(self, text: str):
        if text == "None":
            return None
        if text == "True":
            return True
        if text == "False":
            return False
        if text.startswith(("(", "[", "{")):
            try:
                return ast.literal_eval(text)
            except Exception:
                return text
        return text

    def _stringify_manual_param_value(self, value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (list, tuple, dict)):
            try:
                return json.dumps(value)
            except Exception:
                return str(value)
        return str(value)

    def set_data_train_percent(self, value: str):
        """Set the training percentage, ensuring it is stored as an integer."""
        try:
            self.data_train_percent = int(value)
        except ValueError:
            raise ValueError(f"Invalid value for data_train_percent: {value}. Must be an integer.")
