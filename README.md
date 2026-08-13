# 🚀 AutoMLOps Studio

### Comprehensive Automated Machine Learning & MLOps Platform

[![Version](https://img.shields.io/badge/Version-5.7.1-blue)](https://github.com/PedroM2626/AutoMLOps-Studio)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://automlops-studio.streamlit.app/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Integrated-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

**AutoMLOps Studio** is an end-to-end educational and practical platform designed to simplify the Machine Learning lifecycle. Developed **by a student, for students**, the project provides an intuitive interface to explore everything from data ingestion to production model monitoring, applying the best MLOps principles at every stage.

**🔗 Live Demo:** the platform is deployed on [Streamlit Cloud](https://automlops-studio.streamlit.app/).

> 📖 For the full in-depth documentation, see [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).

---

## 🎯 Objective & Problem Statement

Learning MLOps often requires dealing with complex infrastructures before even understanding the core concepts. This project solves that by centralizing:
- **Unified Workflow**: A clear journey from data upload to deployment across multiple domains (Tabular, Vision, Reinforcement Learning).
- **Visual Experimentation**: Visualize the impact of hyperparameters and architectures in real-time.
- **Production Concepts**: Learn about Data Drift, Model Serving, and Performance Monitoring without the need to configure complex servers.
- **Autonomy**: Train models with automatic tracking of parameters, metrics, models, and dependencies using MLflow.

---

## 🧠 Educational Concepts Built-In

Since AutoMLOps Studio is built for learning, it natively enforces and exposes industry-standard MLOps practices:

### 1. The Tri-Split Rule (Train / Validation / Test)
One of the most confusing concepts for beginners is why data needs to be split multiple times. AutoMLOps enforces a strict, professional evaluation pipeline:
- **Train Split**: The "textbook" the model uses to learn patterns.
- **Validation Split**: The "practice exam". During AutoML, the system tests thousands of hyperparameter combinations and evaluates them here. Since the model is *tuned* based on this score, the result is artificially optimistic.
- **Test Split (Global Holdout)**: The "final exam". This data is isolated entirely at the beginning of the pipeline. The model only sees it **once**, at the very end, providing an unbiased, real-world performance metric free of Data Leakage.

### 2. Preventing Data Leakage
The platform strictly handles preprocessing (like scaling, imputation, or SMOTE) inside Scikit-Learn pipelines. This ensures that transformations are fitted *only* on training data and applied safely to validation/test sets, preventing future information from leaking into the training phase.

### 3. Model Telemetry & Data Drift
Models degrade over time. By serving the model via the FastAPI endpoint, all incoming predictions and inputs are logged to a SQLite telemetry store. The system uses these logs to simulate and detect **Data Drift**, showing students what happens when real-world distributions shift away from the original training data.

---

## 🌟 Key Features

### 1. 🖥️ A Unified GUI with 8 Sections
The entire interface is a single Streamlit application (`app.py`) organized into 8 navigable sections:
**Overview**, **Data**, **AutoML**, **Reinforcement Learning**, **Experiments**, **Registry & Deploy**, **Monitoring**, and **What-If Simulator**.

### 2. 🤖 Multi-Domain Machine Learning
- **Tabular Data**: An Optuna-driven AutoML engine covering Classification, Regression, Forecasting (including LSTM/TCN deep models), Survival Analysis, Uplift Modeling, Clustering, Anomaly Detection, Ranking, Multi-Label & Multi-Task, Association Rules, and Dimensionality Reduction — with presets, early stopping, and time budgets.
- **Computer Vision**: Train models for Image Classification, Image Multi-Label, Semantic Segmentation (DeepLabV3), Object Detection (Faster R-CNN), and Pose Estimation (Keypoint R-CNN), with selectable backbones (ResNet, MobileNetV2, EfficientNet, DenseNet, VGG). `image_anomaly_detection` is also accepted by the CV trainer but is routed through the classification branch.
- **Reinforcement Learning**: A complete module for training agents with **PPO, DQN, A2C, SAC, and TD3** (stable-baselines3) on 9 Gymnasium environments (CartPole, LunarLander, BipedalWalker, …), plus **offline RL via d3rlpy**, custom Gymnasium environment upload, training wrappers, Optuna hyperparameter tuning, and live reward visualization.

### 3. 🧪 Experiments & MLOps Integration
- **MLflow Tracking**: Every experiment run is automatically tracked — configurations, hyperparameters, metrics, and model artifacts (including `rl_config.json`/`rl_config.yaml` for RL runs). Optional remote tracking via DagsHub.
- **Job Manager**: Background training jobs run in separate subprocesses with pause/resume/cancel control, so heavy training never locks the UI.
- **Data Lake**: A versioned, filesystem-backed catalog for datasets and RL agent trajectories (states, actions, rewards, terminal signals).
- **SHAP Explainability**: Model explanations generated via SHAP for trained pipelines.
- **Whitebox Notebook Generation**: The winning AutoML pipeline is automatically exported as a reproducible Jupyter notebook (logged to MLflow as an artifact).
- **5 Pillars of ML**: Every model receives an anatomical profile across the 5 Pillars (see below).

### 4. 📡 Serving, Monitoring & Deployment
- **FastAPI Serving**: Production-ready API (`api.py`) for real-time inference, protected by an API key (`x-api-key` header), with liveness/readiness health endpoints.
- **Live Telemetry**: Every `/predict` call is logged to a SQLite telemetry database for drift and performance analysis.
- **Drift Monitoring**: Statistical drift detection (KS test for numeric features, chi-squared for categorical) plus Deepchecks data-integrity reports.
- **What-If Simulator**: Interactively test models against hypothetical inputs.
- **Deployment Options**: Push models to the Hugging Face Hub, or export a standalone API bundle (FastAPI app + requirements + Dockerfile) as a zip.

---

## 📋 Supported Task Types & Business Objectives

**AutoMLOps Studio** adopts the formal taxonomy from the **Machine Learning Mind Map**, distinguishing between:
1. **Genuine Task Types (Output Structure)**: Strictly defined by the mathematical format of the generated target data (e.g., discrete class label, continuous scalar value, time-to-event pair \((T, E)\), counterfactual vector).
2. **Business Objectives / Applications (Cross-Paradigm)**: Operational business needs that can be addressed through multiple statistical paradigms and different underlying models.

> **Practical Example - Anomaly Detection (`anomaly_detection`)**: It is a **Business Objective**, not a rigid *Task Type*. The platform supports solving it via 4 unsupervised mathematical avenues (plus supervised classification when labels exist):
> - **Spatial Isolation:** `IsolationForest` (random feature splits).
> - **Local Density:** `LocalOutlierFactor` (\(k\)-NN density comparison).
> - **Gaussian Statistical Envelope:** `EllipticEnvelope` (Mahalanobis Distance).
> - **Support Boundary:** `OneClassSVM` (Support hyperplane in Hilbert space).
> - **Supervised Classification:** Via `classification` when historical anomaly labels exist (\(y \in \{0, 1\}\)).

| Modality | Type / Objective | Classification | Brief Description | Main Metrics |
|---|---|---|---|---|
| **Tabular** | `classification` | **Task Type** | Predict a discrete class label (Binary or Multiclass). | `accuracy`, `f1`, `precision`, `recall`, `roc_auc` |
| **Tabular** | `regression` | **Task Type** | Predict a continuous numeric target (Gaussian, Poisson, Gamma GLMs). | `r2`, `rmse`, `mae`, `poisson_deviance`, `gamma_deviance` |
| **Tabular** | `survival_analysis` | **Task Type** | Predict time-to-event with right-censoring \((T, E)\). | `c_index` (Concordance Index) |
| **Tabular** | `uplift_modeling` | **Task Type** | Estimate Individual Treatment Effect (ITE / Causal Inference) via S/T-Learners. | `qini_score` / AUUC |
| **Tabular** | `forecast` | **Objective** | Predict future values from historical temporal data (Lags, Rolling, PyTorch LSTM/TCN). | `r2`, `rmse`, `mae` |
| **Tabular** | `anomaly_detection` | **Objective** | Detect outliers or rare abnormal patterns (IsolationForest, LOF, EllipticEnvelope, OneClassSVM). | `f1`, `decision_score` |
| **Tabular** | `clustering` | **Task Type** | Group samples by similarity without labels. | `silhouette` |
| **Tabular** | `ranking` | **Task Type** | Score items for ordered relevance. | `ndcg` |
| **Tabular** | `multi_label` | **Task Type** | Predict multiple labels per row (multi-target). | `f1_micro`, `subset_accuracy` |
| **Tabular** | `multi_task` | **Task Type** | Predict multiple disparate classification targets concurrently. | `f1_micro`, `subset_accuracy` |
| **Tabular** | `association_rules` | **Task Type** | Discover co-occurrence rules via a custom pairwise rule miner (support / confidence / lift). | `rule_score`, `lift` |
| **Tabular** | `dimensionality_reduction` | **Task Type** | Reduce feature space (PCA, TruncatedSVD, LDA, NCA, PLS). | explained variance / reconstruction |
| **Computer Vision** | `image_classification` | **Task Type** | Assign one class to each image. | `val_acc`, `val_loss` |
| **Computer Vision** | `image_multi_label` | **Task Type** | Assign multiple labels to each image. | `val_acc`, `val_loss` |
| **Computer Vision** | `image_segmentation` | **Task Type** | Pixel-wise semantic segmentation. | `val_score`, `val_loss` |
| **Computer Vision** | `object_detection` | **Task Type** | Detect objects and bounding boxes. | Benchmark metrics |
| **Computer Vision** | `pose_estimation` | **Task Type** | Estimate keypoints/body joints. | Keypoint accuracy |
| **Reinforcement Learning** | `rl_agent` | **Task Type** | Train an agent to maximize reward via PPO, DQN, A2C, SAC, or TD3. | `episode_reward`, `mean_reward` |

---

## 🏛️ Architecture: The 5 Pillars of ML

Every model trained in **AutoMLOps Studio** undergoes an anatomical profile analysis based on the **5 Pillars of ML**:
1. **Pillar 1 (Structure / Skeleton)**: White-box (Linear/Tree/GLM) vs Black-box (Ensemble/Neural Nets).
2. **Pillar 2 (Signal Source)**: Supervised (SL), GLM (Poisson/Gamma), Censored (Survival), Counterfactual (Uplift), Reward (RL), Self-Supervised.
3. **Pillar 3 (Criterion / Loss & Assumed Distribution)**: Mathematical loss derived from the distribution family (Gaussian, Bernoulli, Poisson, Gamma, Cox Likelihood, Qini).
4. **Pillar 4 (Regularization)**: Explicit (L1, L2, ElasticNet, tree max depth) vs Implicit (SGD optimizer bias).
5. **Pillar 5 (Optimizer Engine)**: L-BFGS, Adam, SGD, Optuna TPE, Tree Splitter.

---

## 🆕 What's New (Recent)

- **Temporal & Text Characteristics**: Tabular datasets support "Contains Temporal Data" (automatic chronological validation splits plus lag/rolling-window features) and "Contains Text / NLP Data" (automatic TF-IDF vectorization of text columns).
- **Forecast Task Type**: A dedicated forecasting engine (including LSTM and TCN models implemented in pure PyTorch) integrated across all frameworks.
- **Multi-Task Classification**: Predict multiple target columns concurrently; the interface automatically orchestrates separate training runs per target when needed.
- **Semi-Supervised Learning**: Self-Training classification for targets containing unlabeled samples (`-1` or `NaN`), dynamically wrapping base classifiers in a `SelfTrainingClassifier`.

---

## 📂 Project Structure

```
automlops-studio/
├── app.py                  # Entire Streamlit GUI (design system, 8 sections, wizard pipeline)
├── api.py                  # FastAPI model-serving API (API-key protected, SQLite telemetry)
├── automl_engine.py        # Compatibility facade re-exporting engines for api.py / tests
├── debug_manager.py        # Manual debug script exercising the Job Manager
├── electron-main.js        # Electron desktop wrapper (spawns Streamlit, embeds it)
├── electron-preload.js     # Electron preload (exposes desktop API via contextBridge)
├── src/
│   ├── core/               # Data processor, orchestrator, data lake, drift,
│   │                       #   API-bundle exporter, whitebox notebook generator
│   ├── engines/            # ML engines: classical AutoML, computer vision,
│   │                       #   reinforcement learning, stability analysis,
│   │                       #   PyTorch LSTM/TCN forecasters
│   ├── tracking/           # Job Manager (subprocess workers), MLflow tracking, telemetry
│   ├── deploy/             # Hugging Face Hub deployment helpers
│   └── utils/              # SHAP explainers, 5-Pillars profiles, model cards
├── tests/                  # pytest suite (run with `pytest -q tests/`)
├── data_lake/              # Versioned datasets, RL trajectories, telemetry DB
├── mlruns/                 # MLflow artifacts (metadata in sqlite:///mlflow.db)
├── models/                 # Saved pipelines / RL agents
├── .streamlit/config.toml  # Streamlit server config (headless, port 8501)
├── Dockerfile              # Python 3.13-slim image serving the app on port 7860
├── docker-compose.yml      # 3-service stack: api, dashboard, mlflow
└── requirements.txt        # Fully pinned dependency environment
```

> Note: the GUI lives entirely in `app.py` (a single-file Streamlit app); `src/ui/` is an empty placeholder package.

---

## ⚙️ Setup

The platform targets **Python 3.13** (the Dockerfile and CI both use 3.13; the included devcontainer uses a 3.11 image). The dependency set is heavy (PyTorch and friends — several GB), so installation may take a while.

1. **Install the dependencies**:
```bash
pip install -r requirements.txt
```

2. **Create your environment file** (mandatory):
```bash
copy .env.example .env    # Windows (use `cp` on Linux/macOS)
```
Then edit `.env`. Variables actually used by the code:

| Variable | Required | Description |
|---|---|---|
| `API_SECRET_KEY` | **Yes** | Secret key for the serving API. `api.py` refuses to start without it; clients send it in the `x-api-key` header. |
| `MLFLOW_TRACKING_URI` | No | Defaults to `sqlite:///mlflow.db` (local SQLite backend with artifacts in `mlruns/`). |
| `MLFLOW_TRACKING_USERNAME` | No | Username for remote MLflow tracking (e.g. DagsHub). |
| `MLFLOW_TRACKING_PASSWORD` | No | Password/token for remote MLflow tracking. |

> Note: `LOG_LEVEL`, `MODEL_REGISTRY_PATH`, and `DATA_LAKE_PATH` were removed from `.env.example` because they are not consumed by the code (see [docs/DOCUMENTATION.md §3.3](docs/DOCUMENTATION.md#33-environment-configuration-env)).

### 🐳 Dev Container (GitHub Codespaces friendly)

The repository ships a `.devcontainer/devcontainer.json` based on the Python 3.11 dev container image. It installs the project dependencies automatically on build and launches the Streamlit GUI on port 8501 when the container attaches, so opening the repo in GitHub Codespaces (or VS Code Dev Containers) yields a running app with no local setup.

---

## 🚀 How to Run

### 🐍 Local Streamlit Dashboard
```bash
python -m streamlit run app.py
```
Opens the GUI at `http://localhost:8501`.

### 📡 Serving API
```bash
python api.py
# or
uvicorn api:app --host 0.0.0.0 --port 8000
```
Serves the newest pipeline in `models/` at `http://localhost:8000` (`/`, `/health/live`, `/health/ready`, `/predict`). Requires `API_SECRET_KEY` in `.env`.

### 📊 MLflow UI (optional)
Because the tracking backend is SQLite, pass the backend store explicitly:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlruns --port 5000
```

### 🐳 Docker Compose (full stack)
```bash
docker compose up --build
```
Spins up three services (all read `.env`, so complete the setup step first):
- **Dashboard (Streamlit)**: `http://localhost:8501`
- **Serving API (FastAPI)**: `http://localhost:8000`
- **MLflow UI**: `http://localhost:5000`

### 🤗 Hugging Face Spaces
The project `Dockerfile` follows the HF Spaces convention: the container serves the Streamlit app on port **7860**, which is how the [live Space](https://huggingface.co/spaces/PedroM2626/AutoMLOps-Studio) is deployed. Pushing this repo to a Space (Docker SDK) is all that is required.

### 🖥️ Electron Desktop App
An Electron wrapper bundles the app as a native desktop application:
```bash
npm install
npm start        # runs the app inside Electron
npm run dist     # builds installers via electron-builder (NSIS / AppImage / DMG)
```
Installers for Windows, Linux, and macOS are produced automatically by CI (see below).

---

## 🧪 Testing

The project ships with a pytest suite covering the engines, tracking, data lake, API, and drift logic:
```bash
pytest -q tests/
```

---

## 🔁 CI/CD

Two GitHub Actions workflows keep the project healthy:
- **CI** (`ci.yml`): on every push/PR — installs dependencies on Python 3.13 and runs `pytest -q tests/`.
- **Build Desktop App** (`build-electron.yml`): on pushes to `main`/`master` (and manually) — builds Electron installers on Windows, macOS, and Ubuntu and uploads them as artifacts.

---

## 📜 License

This project is released under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 **Pedro Morato Lahoz**.

---

**Developed by Pedro Morato Lahoz.**
