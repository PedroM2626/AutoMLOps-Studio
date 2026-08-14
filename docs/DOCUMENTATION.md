# AutoMLOps Studio — Project Documentation

> Comprehensive technical reference for the AutoMLOps Studio codebase.
> This documentation reflects the current state of the repository.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Installation & Setup](#3-installation--setup)
4. [Running Guide](#4-running-guide)
5. [Module Reference](#5-module-reference)
   - [src/core](#51-srccore)
   - [src/engines](#52-srcengines)
   - [src/tracking](#53-srctracking)
   - [src/deploy and src/utils](#54-srcdeploy-and-srcutils)
6. [Supported Tasks & Models](#6-supported-tasks--models)
7. [Data & Storage Layout](#7-data--storage-layout)
8. [API Reference](#8-api-reference)
9. [Testing](#9-testing)
10. [CI/CD](#10-cicd)
11. [Deployment](#11-deployment)
12. [Troubleshooting & Known Limitations](#12-troubleshooting--known-limitations)

---

## 1. Overview

AutoMLOps Studio is an end-to-end machine learning operations platform that
combines automated model training (AutoML), reinforcement learning, computer
vision, experiment tracking, model registry, deployment tooling, and production
monitoring into a single desktop/web application.

The project is licensed under the **MIT License** (Copyright (c) 2026 Pedro
Morato Lahoz).

### 1.1 Main Components

| Component | File(s) | Role |
|---|---|---|
| **Streamlit GUI** | `app.py` (~4,900 lines) | The entire graphical application: a single-file Streamlit app with an inline dark-theme CSS design system and a sidebar navigation of **8 sections**: Overview, Data, AutoML, Reinforcement Learning, Experiments, Registry & Deploy, Monitoring, and What-If Simulator. It hosts the wizard-style AutoML pipeline, the data lake browser, drift analysis (Deepchecks with a scipy KS-test fallback), RL training UI, experiment/job dashboard, model registry, deployment actions, and telemetry-based production drift monitoring. |
| **FastAPI serving API** | `api.py` | A lightweight model-serving service (`AutoML Model Serving API`) with health probes and an API-key-protected `/predict` endpoint. It loads the newest `.pkl` pipeline from `models/` and logs every prediction to a SQLite telemetry store. |
| **Engine facade** | `automl_engine.py` | A compatibility facade re-exporting `AutoMLDataProcessor`, `AutoMLTrainer` (an adapter subclass over the classical engine), `RLTrainer`, `TransformersWrapper`, `load_pipeline` / `save_pipeline`, and availability flags. Used by `api.py` and tests. |
| **Electron desktop wrapper** | `electron-main.js`, `electron-preload.js`, `package.json` | Packages the Streamlit app as a desktop application. Spawns a headless Streamlit server as a child process and loads it in a Chromium window. |
| **Docker stack** | `Dockerfile`, `docker-compose.yml` | Containerized deployment: a standalone image for Hugging Face Spaces (port 7860) and a 3-service compose stack (API, dashboard, MLflow UI). |
| **Hugging Face Spaces** | `Dockerfile` | The Dockerfile follows the HF Spaces convention (`EXPOSE 7860`, Streamlit on port 7860) for a public demo deployment. |

Supporting packages:

- `src/core` — data processing, orchestration, data lake, drift detection,
  API bundle export, and reproducible-notebook generation.
- `src/engines` — classical AutoML, computer vision, reinforcement learning,
  stability analysis, and PyTorch time-series models.
- `src/tracking` — background job management, MLflow tracking/registry, and
  inference telemetry.
- `src/deploy`, `src/utils` — Hugging Face deployment, SHAP explainers,
  consumption-code generation, and the "5 Pillars of ML" model profile.
- `src/ui` — an empty placeholder package (both `__init__.py` files are
  0 bytes); the real GUI lives entirely in `app.py`.
- `debug_manager.py` — a manual debug script that submits a dummy
  classification job to `TrainingJobManager` and prints a JSON summary.

---

## 2. Architecture

### 2.1 High-Level Flow

1. The **Streamlit GUI** (`app.py`) collects user configuration (task type,
   dataset, hyperparameter search settings, ensemble options, etc.) through its
   wizard-style interface.
2. Training jobs are submitted to the **`TrainingJobManager`**
   (`src/tracking/manager.py`), which launches each job as an isolated
   **multiprocessing `spawn` worker** subprocess with dedicated log/status
   queues and a pause event.
3. Workers execute the **preprocessing** (`AutoMLDataProcessor`) and the
   selected **engine** (`AutoMLTrainer`, `CVAutoMLTrainer`, or `RLTrainer`).
4. Every run is tracked in **MLflow** using a SQLite backend
   (`sqlite:///mlflow.db`) with artifacts stored in `mlruns/`. Winning
   pipelines can be registered in the MLflow model registry.
5. The **`WhiteboxNotebookGenerator`** produces a reproducible Jupyter notebook
   of the winning pipeline, which is logged to MLflow as an artifact.
6. Registered/exported models can be **deployed** to Hugging Face Hub or
   exported as a standalone FastAPI bundle, or served by the local
   `api.py` from `models/`.
7. In production, `api.py` records every prediction into the
   **`TelemetryStore`** (SQLite, WAL mode). The GUI's Monitoring section reads
   this telemetry to perform **drift monitoring** of incoming inference data.

### 2.2 Diagram

```
                          ┌───────────────────────────────────────────────┐
                          │                Streamlit GUI                  │
                          │  app.py (8 sections: Overview, Data, AutoML,  │
                          │  RL, Experiments, Registry & Deploy,          │
                          │  Monitoring, What-If Simulator)               │
                          └───────┬───────────────────────────▲───────────┘
                                  │ submit / pause / cancel   │ status & logs
                                  ▼                           │
                          ┌───────────────────┐               │
                          │ TrainingJobManager│               │
                          │ (src/tracking/    │               │
                          │  manager.py)      │               │
                          └───────┬───────────┘               │
                                  │ multiprocessing (spawn)   │
                ┌─────────────────┼─────────────────┐         │
                ▼                 ▼                 ▼         │
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
        │ AutoMLTrainer│  │CVAutoMLTrainer│ │  RLTrainer / │  │
        │ (classical)  │  │  (vision)    │  │OfflineRLTrainer│ │
        └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
               │                 │                 │          │
               ▼                 ▼                 ▼          │
        ┌──────────────────────────────────────────────┐      │
        │              MLflow Tracking                 │      │
        │  sqlite:///mlflow.db  +  mlruns/ artifacts   │      │
        │  (optional DagsHub remote tracking)          │      │
        └──────────────────┬───────────────────────────┘      │
                           │ registry                         │
                           ▼                                  │
        ┌──────────────────────────────────────────────┐      │
        │ Registry & Deploy: HF Hub (hf_deploy),       │      │
        │ API bundle export (api_exporter), models/    │      │
        └──────────────────┬───────────────────────────┘      │
                           │ newest *.pkl                     │
                           ▼                                  │
        ┌───────────────────────────────┐    telemetry read   │
        │ FastAPI serving (api.py:8000) │─────────────────────┘
        │  POST /predict (x-api-key)    │
        └──────────────┬────────────────┘
                       │ each prediction
                       ▼
        ┌───────────────────────────────┐
        │ TelemetryStore (SQLite WAL)   │
        │ data_lake/monitoring/         │
        │ telemetry.db (inference_logs) │──► drift monitoring in GUI
        └───────────────────────────────┘
```

### 2.3 Key Design Decisions

- **Process isolation for training**: each job runs in a `spawn`-mode
  subprocess so long trainings never block the GUI and can be paused,
  resumed, or cancelled. A thin entry shim (`src/tracking/worker.py`) exists
  to avoid Windows spawn pickling issues.
- **Local-first MLflow**: the default tracking URI is
  `sqlite:///mlflow.db` with `mlruns/` as the artifact root, so the whole
  platform works offline; DagsHub credentials can optionally point tracking
  to a remote repository.
- **Telemetry loop for monitoring**: rather than an external monitoring
  stack, prediction telemetry is written to SQLite by `api.py` and analyzed
  directly by the GUI's Monitoring section.

---

## 3. Installation & Setup

### 3.1 Requirements

- **Python 3.13** (the Dockerfile base image is `python:3.13-slim` and CI
  runs on Python 3.13). Note: the devcontainer uses a Python **3.11** image.
- Node.js 20 is only required if you build the Electron desktop app.
- Dependencies are fully pinned in `requirements.txt` (a frozen list of
  ~240 packages plus 2 unpinned packages, `nbformat` and `featuretools`).
  Notable pinned versions:

| Package | Version |
|---|---|
| streamlit | 1.54.0 |
| fastapi | 0.115.0 |
| uvicorn | 0.32.0 |
| mlflow | 3.1.0 |
| optuna | 4.7.0 |
| scikit-learn | 1.7.2 |
| xgboost | 3.1.1 |
| lightgbm | 4.6.0 |
| catboost | 1.2.8 |
| torch | 2.9.0 |
| torchvision | 0.24.0 |
| transformers | 4.57.0 |
| stable_baselines3 | 2.7.0 |
| gymnasium | 1.2.3 |
| d3rlpy | 2.3.0 |
| shap | 0.49.1 |
| deepchecks | 0.19.1 |
| pytest | 8.3.4 |

### 3.2 Install Steps

```bash
# 1. Clone the repository and enter it
git clone https://github.com/PedroM2626/AutoMLOps-Studio.git
cd AutoMLOps-Studio

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 3. Install dependencies (large download: includes PyTorch, several GB)
pip install -r requirements.txt
```

### 3.3 Environment Configuration (`.env`)

Copy the example file and fill in the values:

```bash
cp .env.example .env      # Windows (PowerShell): Copy-Item .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `API_SECRET_KEY` | **Required by `api.py`** | Shared secret for the serving API. The FastAPI app raises a startup error if it is missing, and `/predict` validates the `x-api-key` header against it. |
| `MLFLOW_TRACKING_URI` | Optional | Defaults to `sqlite:///mlflow.db` when unset. |
| `MLFLOW_TRACKING_USERNAME` | Optional | DagsHub username for remote MLflow tracking. |
| `MLFLOW_TRACKING_PASSWORD` | Optional | DagsHub token/password for remote MLflow tracking. |
| `LOG_LEVEL` | Unsupported | Removed from `.env.example`; not read anywhere in the code. |
| `MODEL_REGISTRY_PATH` | Unsupported | Removed from `.env.example`; the registry path is hardcoded to `models/`. |
| `DATA_LAKE_PATH` | Unsupported | Removed from `.env.example`; the data lake path is hardcoded to `./data_lake`. |
| `IS_ELECTRON_APP` | Automatic | Set automatically by `electron-main.js`; do not set manually. |

### 3.4 Streamlit Configuration

`.streamlit/config.toml` configures the server:

- `[server]`: `headless = true`, `enableCORS = false`, `port = 8501`
- `[browser]`: `gatherUsageStats = false`

### 3.5 Devcontainer

`.devcontainer/devcontainer.json` provides a GitHub Codespaces /
VS Code devcontainer based on a **Python 3.11** (bookworm) image. It installs
`requirements.txt` and starts Streamlit on port 8501 with
`--server.enableCORS false --server.enableXsrfProtection false`.

---

## 4. Running Guide

### 4.1 Streamlit GUI (local)

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501**. The sidebar provides the 8 navigation
sections: Overview, Data, AutoML, Reinforcement Learning, Experiments,
Registry & Deploy, Monitoring, and What-If Simulator.

### 4.2 Model Serving API (local)

```bash
# Requires API_SECRET_KEY in .env
python api.py
# or
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for the interactive OpenAPI docs.
Authentication for `POST /predict` uses the `x-api-key` header compared
against `API_SECRET_KEY`. See [Section 8](#8-api-reference) for the full
endpoint reference and a curl example.

### 4.3 MLflow UI

The default tracking URI is `sqlite:///mlflow.db` with artifacts under
`mlruns/`, so launch the UI with an explicit backend store:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlruns --port 5000
```

Open **http://localhost:5000**.

### 4.4 Docker Compose Stack

`docker-compose.yml` defines three services, all mounting the repository at
`/app`, using `env_file: .env`, and restarting unless stopped:

| Service | Command / Image | Container port | Host port |
|---|---|---|---|
| `api` | `uvicorn api:app --host 0.0.0.0 --port 8000` | 8000 | **8000** |
| `dashboard` | Streamlit (`app.py`), depends on `api` | 8501 | **8501** |
| `mlflow` | image `ghcr.io/mlflow/mlflow:v2.19.0`, `mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////app/mlflow.db --default-artifact-root /app/mlruns` | 5000 | **5000** |

```bash
# Ensure .env exists (the api service fails without API_SECRET_KEY)
docker compose up --build
```

Then open http://localhost:8501 (GUI), http://localhost:8000 (API),
and http://localhost:5000 (MLflow UI).

### 4.5 Hugging Face Spaces

The root `Dockerfile` follows the HF Spaces Docker convention:
`EXPOSE 7860` and a CMD that starts Streamlit on port **7860**. Pushing the
repository to a Space of type "Docker" runs the GUI at the Space URL.

### 4.6 Electron Desktop App

```bash
npm install     # install electron, electron-builder, wait-on, tree-kill
npm start       # run the desktop app (electron .)
npm run dist    # build distributable packages (electron-builder, --publish never)
```

How it works: `electron-main.js` spawns
`python -m streamlit run app.py --server.port 8501 --server.headless true`
(preferring `venv/Scripts/python.exe` when present), waits for the server
with up to 20 retries, loads `http://127.0.0.1:8501` in a BrowserWindow
(falling back to `error_loading.html`), sets `IS_ELECTRON_APP=true`, and
kills the Python process on quit. `electron-preload.js` exposes
`window.electronAPI = { isDesktop: true, platform }` via contextBridge.

Build targets per platform (configured in `package.json`, `asar: false`):

| Platform | Target |
|---|---|
| Windows | `nsis` installer (`.exe`) |
| Linux | `AppImage` |
| macOS | `dmg` |

Note: the build references `assets/**` and `assets/icon.png`; the icon path
is guarded by an `existsSync` check, and the `assets/` directory is not
present in the repository. No code signing is configured.

---

## 5. Module Reference

### 5.1 `src/core`

#### `processor.py` — `AutoMLDataProcessor`

- **Purpose:** end-to-end tabular preprocessing built on sklearn
  `ColumnTransformer` / `Pipeline`.
- **Constructor:** `AutoMLDataProcessor(target_column, task_type, data_type,
  date_col, forecast_horizon, nlp_config, scaler_type, semi_supervised,
  enable_dfs, dfs_depth, impute_strategy, impute_fill_value, encoding_mode,
  onehot_cardinality_threshold, clip_outliers, outlier_lower_q,
  outlier_upper_q, ts_clustering_config)`.
- **Methods:** `fit_transform(df, nlp_cols)` and `transform(df)`.
- **Capabilities:** configurable scalers (`auto`/`standard`, `none`,
  `minmax`, `robust`, `maxabs`, `quantile`, `power` — see
  `build_scaler()`), OneHot/Ordinal encoders with selectable encoding
  mode and cardinality threshold, selectable imputation strategy,
  optional Winsorization (`Winsorizer`); Deepchecks data-integrity HTML
  report; temporal feature engineering (hour, day-of-week, etc., plus
  lags and rolling mean/std — including lags derived from encoded
  categorical targets for Forecast/TS Classification); **TS Clustering
  windowing** (`_apply_ts_windows`: sliding-window summary features);
  configurable **NLP text vectorization** (`nlp_config`: TF-IDF,
  Bag-of-Words counts, binary Bag-of-Words, feature hashing, contextual
  embeddings, or raw passthrough; cleaning modes standard / god_mode /
  none; n-gram range, vocabulary cap, stop words); semi-supervised
  label handling
  (`-1` / NaN treated as unlabeled); optional featuretools DFS.

#### `trainer.py` — `TransformersWrapper`

- **Purpose:** fine-tunes Hugging Face transformers
  (BERT / DistilBERT / RoBERTa / DeBERTa) for text classification/regression
  via PyTorch, exposing an sklearn-compatible `fit` / `predict` /
  `predict_proba` interface.
- **Constructor:** `TransformersWrapper(model_name, task, epochs, learning_rate)`.
- **Helpers:** `get_ensemble_display_name()`; model-key sets
  `_ENSEMBLE_MODEL_KEYS` and `_DL_MODEL_KEYS`. Falls back gracefully when
  given vectorized (non-text) input.

#### `orchestrator.py` — `AutoMLOrchestrator`

- **Purpose:** UI-independent facade over the three engine families.
- **Constructor:** `AutoMLOrchestrator(config)`.
- **Methods:** `submit_classical_job(job_manager)`, `run_vision_training(...)`,
  `run_rl_training()`.
- **Artifacts:** RL results are saved to `models/rl/rl_agent_{env}_{algo}`.

#### `data_lake.py` — `DataLake`

- **Purpose:** filesystem-backed versioned dataset catalog.
- **Constructor:** `DataLake(base_path="./data_lake")`.
- **Methods:** `list_datasets`, `list_versions`, `get_version_info`,
  `load_version` (csv / json / parquet / txt / zip), `save_raw_file`,
  `save_dataframe`, `delete_version`; versions are stored as
  `v_YYYYMMDD_HHMMSS.<ext>` and guarded against path traversal.
  `DatasetVersion` is the accompanying dataclass.

#### `drift.py` — `DriftDetector`

- **Purpose:** statistical drift detection.
- **API:** `DriftDetector.detect_drift(reference_data, current_data, threshold=0.05)`
  — KS test for numeric columns, chi-squared for categorical columns.
- **Note:** currently consumed only by tests; the GUI drift tabs implement
  their own inline KS logic plus Deepchecks.

#### `api_exporter.py` — `export_model_api`

- **Purpose:** standalone API bundle export.
- **API:** `export_model_api(model_name, version) -> zip path`. Pulls a
  registered model from MLflow and bundles a self-contained FastAPI app,
  a requirements file, and a Dockerfile (`python:3.10-slim`, port 8000)
  into a zip archive. Wired into the Registry & Deploy section of the GUI.

#### `notebook_generator.py` — `WhiteboxNotebookGenerator`

- **Purpose:** reproducibility/white-box output of the winning pipeline.
- **API:** `WhiteboxNotebookGenerator(config, best_params, feature_names,
  dataset_path).generate()` produces a Jupyter notebook (including the
  5-Pillars profile). The notebook is written to the current directory as
  `automl_candidate_pipeline_<hex>.ipynb` and logged to MLflow as an
  artifact by the job worker.

### 5.2 `src/engines`

#### `classical.py` — `AutoMLTrainer`

The core AutoML engine (~3,300 lines).

- **Constructor:** `AutoMLTrainer(task_type, preset, ensemble_config,
  use_ensemble, use_deep_learning, ensemble_mode, n_jobs, data_type,
  semi_supervised)`.
- **Search:** Optuna-driven hyperparameter optimization supporting
  `bayesian`, `random`, and `grid` modes, with presets
  `fast` / `medium` / `test` (the GUI additionally offers `high` and
  `custom`), early stopping, time budgets, and a
  `stability_config`.
- **Ensembles:** `voting` and `stacking` modes.
- **Semi-supervised:** wraps base estimators with `SelfTrainingClassifier`.
- **Additional API:** `get_technical_explanation()`,
  `save_pipeline()` / `load_pipeline()` (joblib), SHAP plots via
  `ModelExplainer`.

Task types: `classification`, `regression`, `forecast` (including `lstm`
and `tcn` via `PyTorchTimeSeriesRegressor`), `survival_analysis`
(c-index metric, custom `calculate_c_index`), `uplift_modeling`
(s-learner / t-learner, custom `calculate_qini_score`), `clustering`,
`anomaly_detection`, `multi_label`, `multi_task`, `multi_regression`
(multi-output/multivariate regression via `MultiOutputRegressor`),
`ranking`,
`association_rules` (custom pairwise `AssociationRuleMiner` with
support/confidence/lift), and `dimensionality_reduction`.
See [Section 6](#6-supported-tasks--models) for the consolidated model list.

#### `vision.py` — `CVAutoMLTrainer`

- **Constructor:** `CVAutoMLTrainer(task_type='image_classification',
  num_classes=2, backbone='resnet18', multilabel_threshold=0.5)`.
- **Tasks:** `image_classification`, `image_multi_label`
  (`MultiLabelImageDataset` + `label_csv`), `image_segmentation`
  (DeepLabV3-ResNet50 + `mask_dir`), `object_detection`
  (Faster R-CNN ResNet50-FPN), `pose_estimation` (Keypoint R-CNN).
  `image_anomaly_detection` is also accepted but is routed through the
  classification branch.
- **Backbones:** resnet18, resnet50, mobilenet_v2, efficientnet_b0,
  densenet121, vgg16.
- **Options:** augmentation config; optimizers adam / sgd / rmsprop.
- **Helper:** `get_cv_explanation()`.

#### `reinforcement_learning.py`

- **`RLTrainer`:** trains agents with PPO, DQN, A2C, SAC, or TD3
  (`RLTrainer.ALGORITHMS`), with default hyperparameters plus Optuna search
  spaces, `Monitor` / `VecNormalize` wrappers, custom Gymnasium environment
  loading, early stopping on reward threshold, and checkpointing to
  `tmp/checkpoints`. Logs `rl_config.json` / `rl_config.yaml` artifacts.
- **`StreamlitRLCallback`:** live MLflow metric logging
  (`episode_reward`, `episode_length`, `mean_reward_100`) and trajectory
  capture into the DataLake as `rl_trajectories_<env>`.
- **`OfflineRLTrainer`:** offline RL via d3rlpy.
- **`compare_agents()`:** agent comparison utility.
- **`get_available_rl_environments()`:** 9 built-in Gymnasium environments —
  CartPole-v1, MountainCar-v0, MountainCarContinuous-v0, Pendulum-v1,
  Acrobot-v1, LunarLander-v2, LunarLanderContinuous-v2, BipedalWalker-v3,
  BipedalWalkerHardcore-v3. Custom environments can be uploaded through the
  GUI (stored under `tmp/`).

#### `stability.py` — `StabilityAnalyzer`

- **Constructor:** `StabilityAnalyzer(base_model, X, y, task_type)`.
- **Analyses:** seed stability, data-split stability, and cross-validation
  (KFold / StratifiedKFold / TimeSeriesSplit). Used by the Monitoring
  section ("Model Robustness & Stability").

#### `pytorch_forecast.py` — custom PyTorch forecasters

Despite the filename, this module **does not use the pytorch-forecasting
library**. It implements pure PyTorch models — `LSTMForecaster` and
`TCNForecaster` (with `Chomp1d` / `TemporalBlock`) — wrapped as an sklearn
estimator `PyTorchTimeSeriesRegressor(model_type='lstm' | 'tcn')`, which the
classical engine uses for the `forecast` task.

`src/engines/__init__.py` re-exports the trainers and sets
`CVAutoMLTrainer = None` as a placeholder (vision is imported lazily by the
orchestrator/app).

### 5.3 `src/tracking`

#### `manager.py` — `TrainingJobManager`

- **`JobStatus` enum:** `queued`, `running`, `paused`, `completed`,
  `failed`, `cancelled`.
- **`TrainingJob`:** dataclass describing a job.
- **`TrainingJobManager`:** `submit_job`, `pause_job`, `resume_job`,
  `cancel_job`, `delete_job`, `poll_updates`, `has_running_jobs`, `get_job`.
  Each job is a multiprocessing **spawn** subprocess with log/status queues
  and a pause event.
- **Worker behavior:** `_training_worker` sets the MLflow tracking URI
  (default `sqlite:///mlflow.db`), applies optional DagsHub credentials, runs
  preprocessing + `AutoMLTrainer`, evaluates on a holdout set, generates the
  whitebox notebook, and emits a results dictionary.

#### `worker.py`

Thin `training_worker_entry()` shim used as the subprocess entry point to
avoid Windows spawn pickling issues.

#### `mlflow.py` — `MLFlowTracker` and registry helpers

- **`MLFlowTracker(experiment_name).log_experiment(...)`:** logs params,
  metrics, and the sklearn model, plus an auto-generated consumption-code
  artifact.
- **Registry helpers:** `get_all_runs`, `get_model_registry`,
  `register_model_from_run`, `get_registered_models`, `get_model_details`,
  `load_registered_model`, `get_run_details`, `get_model_signature`.
- Includes a `RunInfo` `run_uuid` / `run_id` monkeypatch for compatibility.

#### `telemetry.py` — `TelemetryStore`

- **Constructor:** `TelemetryStore(db_path="data_lake/monitoring/telemetry.db")`.
- **Storage:** SQLite in WAL mode; table `inference_logs` with columns
  `timestamp_utc`, `model_version`, `row_count`, `request_json`,
  `predictions_json`.
- **Writer:** `api.py` logs every `/predict` call via `log_inference`.

### 5.4 `src/deploy` and `src/utils`

#### `deploy/hf_deploy.py` — `deploy_to_huggingface`

`deploy_to_huggingface(model_path, repo_id, token, private,
model_card_data)` creates the repository, uploads files/folders, and
generates a model card. A stub `test_inference_hf()` is also provided.
Used from the Registry & Deploy section of the GUI.

#### `utils/explainers.py` — `ModelExplainer`

SHAP-based model explanations, used by `classical.py` to produce SHAP plots.

#### `utils/helpers.py`

- `get_consumption_code()` / `get_cv_consumption_code()`: generate
  ready-to-use model consumption snippets (embedded into MLflow artifacts).
- `generate_model_card()`: model card generation.
- Embeds the default tracking URI
  `MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")`
  into generated code.

#### `utils/pillars.py`

`get_model_pillars_profile(model_name, task_type, params)` implements the
"5 Pillars of ML" profile used by the whitebox notebook generator and tests.

---

## 6. Supported Tasks & Models

Consolidated list of task types handled by `AutoMLTrainer` (classical
engine) and the specialized engines:

| Task type | Engine | Models / algorithms | Metric notes |
|---|---|---|---|
| `classification` | classical | logistic_regression, random_forest, xgboost, lightgbm, extra_trees, decision_tree, svm, linear_svc, knn, mlp, sgd_classifier, passive_aggressive, naive_bayes, ridge_classifier, adaboost, bagging, hist_gradient_boosting, catboost; transformers (bert, distilbert, roberta, deberta); ensembles (voting / stacking) | Also used for **Time-Series Classification** when `data_type='sequential'` (chronological holdout + `TimeSeriesSplit` CV) |
| `regression` | classical | linear_regression, random_forest, xgboost, lightgbm, extra_trees, decision_tree, svm, knn, mlp, ridge, lasso, elastic_net, sgd_regressor, adaboost, bagging, poisson, gamma (GLMs), hist_gradient_boosting, catboost; transformers (-reg variants for **NLP regression**: bert-base-uncased-reg, distilbert-base-uncased-reg) | — |
| `forecast` | classical + pytorch_forecast | random_forest, xgboost, extra_trees, catboost, **lstm**, **tcn** (via `PyTorchTimeSeriesRegressor`) | `r2`, `rmse`, `mae`, `mape` |
| `forecast_classification` | classical | Full classification catalog; internally mapped to `classification` with forced temporal behavior: lag features derived from the (encoded) categorical target, chronological holdout splits, and `TimeSeriesSplit` validation | `accuracy`, `f1`, `precision`, `recall`, `roc_auc` |
| `survival_analysis` | classical | survival estimators | c-index (`calculate_c_index`) |
| `uplift_modeling` | classical | s_learner, t_learner | Qini score (`calculate_qini_score`) |
| `clustering` | classical | kmeans, agglomerative, dbscan, gaussian_mixture, mean_shift, birch, spectral | `silhouette`, `calinski_harabasz`, `davies_bouldin` |
| `ts_clustering` | classical (processor windowing) | Same clustering catalog. `AutoMLDataProcessor` segments the chosen numeric series into sliding windows (configurable `window_size` / `step`) and extracts per-window summary features (`mean`, `std`, `min`, `max`, `median`, `skew`, `trend`) before clustering | `silhouette` |
| `anomaly_detection` | classical | **11 detectors:** isolation_forest, local_outlier_factor, elliptic_envelope, one_class_svm, zscore_detector, modified_zscore (MAD), mahalanobis (empirical or robust MCD covariance), hbos (Histogram-Based Outlier Score), knn_outlier, pca_residual, rolling_residual (time-series oriented) | `decision_score`; semi-supervised `f1`/`precision`/`recall` when labels (1 = anomaly) are provided |
| `density_estimation` | classical | kernel_density (`KernelDensity` wrapper: bandwidth + kernel search), gaussian_mixture_density (`GaussianMixture` wrapper), histogram_density (independent per-feature histograms) | Held-out `log_likelihood` (maximized); high-dimensional NLP features are first projected with TruncatedSVD |
| `multi_label` | classical | MultiOutputClassifier wrappers over base classifiers | — |
| `multi_task` | classical | MultiOutputClassifier orchestration | — |
| `multi_regression` | classical | Full regression catalog (linear/ridge/lasso/elastic-net, random_forest, xgboost, lightgbm, extra_trees, svr, knn, mlp, …) wrapped with `MultiOutputRegressor` for simultaneous prediction of several continuous targets | `r2`, `rmse`, `mae`, `mape` (uniform average across outputs; `evaluate` also reports `n_outputs` and `per_output_r2`) |
| `ranking` | classical | ranking-aware training | — |
| `association_rules` | classical | custom pairwise `AssociationRuleMiner` (support / confidence / lift) — **not** Apriori/FP-Growth | — |
| `dimensionality_reduction` | classical | pca, truncated_svd, lda, nca, pls | — |
| `image_classification` | vision | backbone CNNs (resnet18/50, mobilenet_v2, efficientnet_b0, densenet121, vgg16) | — |
| `image_multi_label` | vision | backbone CNNs with `MultiLabelImageDataset` | — |
| `image_segmentation` | vision | DeepLabV3-ResNet50 | — |
| `object_detection` | vision | Faster R-CNN ResNet50-FPN | — |
| `pose_estimation` | vision | Keypoint R-CNN | — |
| Reinforcement learning | RL engine | PPO, DQN, A2C, SAC, TD3 (online, stable-baselines3); d3rlpy algorithms (offline) | episode reward metrics |

Additional cross-cutting capabilities:

- **Hyperparameter search:** Optuna bayesian / random / grid modes with
  `fast`, `medium`, and `test` presets (the GUI additionally offers
  `high` and `custom`), early stopping and time budgets.
- **Ensembles:** voting and stacking.
- **Semi-supervised learning:** `SelfTrainingClassifier` wrapping for
  partially labeled data.
- **Text NLP:** selectable text vectorization (TF-IDF, Bag-of-Words,
  binary Bag-of-Words, feature hashing, Sentence-Transformer embeddings
  or raw-text passthrough) and transformer fine-tuning
  (classification *and* regression heads).
- **Customizable preprocessing** (`AutoMLDataProcessor`):
  - *Feature scaling* (`scaler_type`): `auto` / `standard` (default),
    `none`, `minmax`, `robust`, `maxabs`, `quantile`, `power`
    (Yeo-Johnson). Exposed through `build_scaler()` with sparse-safe
    fallbacks.
  - *Imputation* (`impute_strategy`): `median` (default), `mean`,
    `most_frequent`, `constant` (`impute_fill_value`).
  - *Categorical encoding* (`encoding_mode`): `auto` (one-hot for low
    cardinality, ordinal for high cardinality), `onehot`, or `ordinal`,
    with a configurable `onehot_cardinality_threshold`.
  - *Outlier treatment*: optional Winsorization (`clip_outliers`) that
    clips numeric features to `[outlier_lower_q, outlier_upper_q]`
    quantile bounds before scaling.
  - *Per-model scaling overrides*: `train(..., scaler_overrides={model:
    scaler})` wraps individual models in a
    `Pipeline(model_scaler -> model)` so each algorithm can receive a
    different scaling of the same preprocessed features (also available
    in the GUI Model Selection step).
  - *NLP text preprocessing* (`nlp_config`, GUI panel "📝 NLP Text
    Preprocessing" in wizard Step 5):
    - `vectorizer`: `tfidf` (default), `count` (Bag-of-Words with raw
      term counts), `binary` (Bag-of-Words presence flags), `hashing`
      (stateless feature hashing with fixed width — suited to huge
      vocabularies), `embeddings` (Sentence-Transformers, configurable
      via `embedding_model`, falls back to TF-IDF if unavailable), or
      `passthrough` (raw text kept for Transformer models).
    - `cleaning_mode`: `standard` (lowercase, strip URLs/mentions/
      punctuation), `god_mode` (aggressive normalization + accent
      stripping), or `none` (raw text untouched).
    - `ngram_range` (e.g. `(1,1)`, `(1,2)`, `(1,3)`, `(2,2)`),
      `max_features` vocabulary cap (auto-reduced when several text
      columns are present), `stop_words` toggle with `language`
      selection, and `sublinear_tf` for TF-IDF.

---

## 7. Data & Storage Layout

| Path | Written by | Content |
|---|---|---|
| `data_lake/<dataset>/v_YYYYMMDD_HHMMSS.<ext>` | `DataLake.save_raw_file` / `save_dataframe` via the GUI upload UI | Versioned user datasets |
| `data_lake/rl_trajectories_<env>/` | `StreamlitRLCallback` (training end) | Saved RL trajectories (CSV) |
| `data_lake/monitoring/telemetry.db` | `TelemetryStore` (api.py `/predict`) | SQLite inference logs (WAL mode) |
| `data_lake/uploads/*.csv` | Legacy artifacts | Hash-prefixed files not producible by current code (stale demo data) |
| `data_lake/processed_train/`, `data_lake/processed_validation/` | Legacy artifacts | Datasets from a prior pipeline version |
| `mlruns/` + `mlflow.db` | MLflow (tracking URI `sqlite:///mlflow.db`, artifact root `mlruns/`) | Experiment metadata and artifacts; RL runs include `rl_config.json` / `rl_config.yaml` |
| `models/` | RL orchestrator saves `models/rl/rl_agent_{env}_{algo}`; `api.py` reads the **newest `*.pkl`** via `load_pipeline` (joblib) | Deployable model artifacts (empty in a fresh checkout) |
| `tmp/best_model`, `tmp/checkpoints` | RL callbacks (EvalCallback / CheckpointCallback) | RL checkpoints |
| `tmp/eval_logs/evaluations.npz` | RL evaluation logging | Evaluation curves |
| `tmp/<custom_env>.py` | GUI custom Gymnasium environment upload | User-defined RL environments |
| `catboost_info/` | CatBoost training side-effect | Training logs (candidate for gitignore) |
| `automl_candidate_pipeline_*.ipynb` | `WhiteboxNotebookGenerator` (CWD), then logged to MLflow | Reproducible winning-pipeline notebooks |

---

## 8. API Reference

The serving API is defined in `api.py` as
`app = FastAPI(title="AutoML Model Serving API")`.

**Authentication:** `POST /predict` requires the header `x-api-key` whose
value must equal the `API_SECRET_KEY` environment variable. Wrong keys
return `403 Invalid API Key`; if the server has no key configured, requests
receive `503`.

**Model loading:** at startup (and lazily on first `/predict`), the newest
`.pkl` file in `models/` is loaded via `load_pipeline` into a processor +
model pair.

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/` | none | Returns `{"status": "online", "model_loaded": <bool>}` |
| `GET` | `/health/live` | none | Liveness probe; returns `{"status": "alive"}` |
| `GET` | `/health/ready` | none | Readiness probe; `200` with model name when the API key is configured and a model is loaded, otherwise `503` with a detail object (`has_api_key`, `model_loaded`) |
| `POST` | `/predict` | `x-api-key` header | Accepts `{"data": [ {row}, ... ]}`, transforms rows through the processor, predicts, inverse-transforms classifier labels when available, logs the call to telemetry, and returns `{"predictions": [...]}`. Returns `400` if no model is available, `500` on processing errors. |

### Example: scoring a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_SECRET_KEY" \
  -d '{
    "data": [
      {"crim": 0.00632, "zn": 18.0, "indus": 2.31, "rooms": 6.575, "price": 24.0}
    ]
  }'
```

Response:

```json
{"predictions": [24.1]}
```

Rows must match the columns the winning pipeline was trained on (the
processor expects raw, pre-transformation features, including the target
column layout used at training time).

---

## 9. Testing

There is no pytest configuration file (no `pytest.ini` / `pyproject.toml` /
`setup.cfg`). Run the suite with:

```bash
pytest -q tests/
# or the provided wrapper (runs pytest.main(["tests", "-v"])):
python tests/run_tests.py
```

`tests/conftest.py` provides an autouse fixture that redirects MLflow to a
temporary `file:///` store for every test, keeping the real
`sqlite:///mlflow.db` untouched.

### Test coverage by file

| Test file | Coverage |
|---|---|
| `test_core.py` | processor, AutoMLTrainer classification/regression, DriftDetector, save/load_pipeline |
| `test_classical.py` | classical model zoo, supervised dimensionality reduction |
| `test_mapa_mental.py` | 5-pillars profile, model card, GLM Poisson/Gamma, survival, uplift |
| `test_pytorch_forecast.py` | LSTM/TCN regressors |
| `test_rl.py` | RLTrainer init/train/save/load/evaluate, environments, wrappers |
| `test_orchestrator.py` | classical submission + RL training via orchestrator |
| `test_api.py` | `/health/live`, API-key enforcement, predict + SQLite telemetry |
| `test_data_lake.py` | save/load/delete + path-traversal rejection |
| `test_drift.py` | numeric/categorical drift signals |
| `test_mlflow_tracking.py` | `register_model_from_run` artifact resolution |
| `test_patch.py` | RunInfo `run_uuid` patch |
| `test_stability_integration.py` | StabilityAnalyzer |
| `test_card.py` | model card generation |
| `test_streamlit_gui.py` | app imports / initialization |
| `test_reflex_services.py` | asserts the legacy Reflex module is gone |
| `test_interface_simulation_unified.py` | unittest-based unified interface simulation |
| `inspect_runinfo.py`, `reproduce_mlflow_error.py` | non-pytest debug scripts |

---

## 10. CI/CD

### 10.1 `ci.yml` ("CI")

- **Triggers:** push and pull requests.
- **Runner:** `ubuntu-latest`, 40-minute job timeout.
- **Steps:** set up Python **3.13** with pip caching,
  `pip install -r requirements.txt`, then run `pytest -q tests/` with the
  environment variable `API_SECRET_KEY=ci-test-key` (required by API tests).

### 10.2 `build-electron.yml` ("Build Desktop App")

- **Triggers:** pushes to `main` / `master` plus `workflow_dispatch`.
- **Matrix:** windows-latest, macos-latest, ubuntu-latest.
- **Steps:** Node 20 with npm cache, `npm install`, then
  `npm run dist -- --win` / `--mac` / `--linux` (`GH_TOKEN` provided on
  macOS).
- **Artifacts:** uploads `dist/*.exe`, `dist/*.dmg`, or `dist/*.AppImage`
  named `automlops-studio-<os>` with 7-day retention.
- No code signing is configured.

---

## 11. Deployment

### 11.1 Docker (single image)

The root `Dockerfile`:

- Base image `python:3.13-slim`.
- Installs system packages `build-essential`, `libgomp1`, `libgl1`,
  `libglib2.0-0`, and `curl`.
- Installs `requirements.txt`, runs as a non-root user (uid/gid 1000), and
  creates `mlruns/`, `data_lake/`, and `models/`.
- `EXPOSE 7860`; CMD runs Streamlit on port 7860 (Hugging Face Spaces
  convention).

```bash
docker build -t automlops-studio .
docker run -p 7860:7860 --env-file .env automlops-studio
```

### 11.2 Docker Compose

See [Section 4.4](#44-docker-compose-stack) for the full service table
(api on 8000, dashboard on 8501, MLflow UI on 5000).

### 11.3 Hugging Face Spaces

Push the repository to a Space configured with the Docker SDK. The
Dockerfile's port 7860 entrypoint matches the Spaces convention, so the GUI
becomes publicly available at the Space URL (note the public Space sleeps
after inactivity).

### 11.4 Hugging Face Hub model deployment (from the GUI)

The Registry & Deploy section calls `deploy_to_huggingface(model_path,
repo_id, token, private, model_card_data)` from `src/deploy/hf_deploy.py`,
which creates the Hub repository, uploads the model files, and generates a
model card.

### 11.5 Standalone API bundle export

`export_model_api(model_name, version)` (Registry & Deploy section) pulls a
registered model from MLflow and produces a zip containing:

- a standalone FastAPI serving app,
- a `requirements.txt`,
- a `Dockerfile` (base `python:3.10-slim`, port 8000).

Unzip the bundle and run it independently of the Studio.

---

## 12. Troubleshooting & Known Limitations

### 12.1 Common issues

| Symptom | Cause | Resolution |
|---|---|---|
| API fails to start with `RuntimeError: Missing required environment variable API_SECRET_KEY` | `API_SECRET_KEY` is unset | Create `.env` from `.env.example` and set `API_SECRET_KEY` before starting `api.py` |
| `/predict` returns `400 No model loaded. Train a model first.` | No `.pkl` pipeline in `models/` | Train and save a pipeline first (joblib `.pkl` in `models/`) |
| `/health/ready` returns `503` | Missing API key or no model loaded | Check the detail object's `has_api_key` / `model_loaded` flags |
| `mlflow ui` shows no experiments | Launched without the backend store URI | Use `mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlruns --port 5000` |
| Very slow / huge install | `requirements.txt` pins ~240 packages including PyTorch | Expected; allow several GB of disk and a stable connection |
| Electron window shows the error page | Streamlit server not up after 20 retries | Ensure the Python environment (or `venv/Scripts/python.exe`) can run Streamlit; check port 8501 availability |

### 12.2 Known limitations

- **Heavy dependency footprint:** the frozen requirements include PyTorch,
  transformers, and the full ML stack; installation is several gigabytes.
- **Unused environment variables removed:** `LOG_LEVEL`, `MODEL_REGISTRY_PATH`,
  and `DATA_LAKE_PATH` were removed from `.env.example` because no code
  consumes them; the paths are hardcoded (`models/`, `./data_lake`).
- **`pytorch_forecast.py` naming:** the module does not depend on the
  pytorch-forecasting library; it implements custom PyTorch LSTM/TCN
  forecasters.
- **Association rules:** implemented as a custom pairwise rule miner
  (support/confidence/lift), not Apriori/FP-Growth.
- **Legacy data lake files:** `data_lake/uploads/` hash-prefixed CSVs and
  the `processed_train` / `processed_validation` versions are artifacts from
  older application versions and cannot be reproduced by current code.
- **Empty `src/ui` package:** the GUI is a single-file Streamlit app
  (`app.py`); `src/ui` is a placeholder.
- **Missing desktop assets:** the Electron build references `assets/`
  (including an app icon), which is absent from the repository; the icon
  path is guarded so builds still succeed.
- **This documentation reflects the current state of the codebase**; future
  changes may render parts outdated.

---

*Developed by Pedro Morato Lahoz. Licensed under the MIT License.*
