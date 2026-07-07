# 🚀 AutoMLOps Studio

### Comprehensive Automated Machine Learning & MLOps Platform

[![Version](https://img.shields.io/badge/Version-v5.5.0-blue)](https://github.com/PedroM2626/automlops-studio)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PedroM2626/AutoMLOps-Studio)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Integrated-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

**AutoMLOps Studio** is an "end-to-end" educational and practical platform designed to simplify the Machine Learning lifecycle. Developed **by a student, for students**, the project provides an intuitive interface to explore everything from data ingestion to production model monitoring, applying the best MLOps principles at every stage.

**🔗 Access the live Demo:** [Streamlit Cloud - AutoMLOps Studio](https://automlops-studio.streamlit.app/)

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
Models degrade over time. By serving the model via the FastAPI endpoint, all incoming predictions and actual inputs are logged. The system uses these logs to simulate and detect **Data Drift**, showing students what happens when real-world distributions shift away from the original training data.

---

## 🌟 Key Features

### 1. 🤖 Multi-Domain Machine Learning
- **Tabular Data**: Full support for Classification, Regression, Time Series Forecasting, Clustering, Anomaly Detection, Ranking, Multi-Label, and Association Rules. Includes Optuna optimization.
- **Computer Vision**: Train models for Image Classification, Multi-Label, Segmentation, Object Detection, Anomaly Detection, and Pose Estimation with the same MLOps pipeline as tabular data.
- **Reinforcement Learning**: Complete module for training agents (PPO, DQN, A2C, SAC, TD3) on environments like CartPole, LunarLander, etc. Includes online training, offline RL (via d3rlpy), custom Gymnasium environments, wrappers, Optuna hyperparameter tuning, and live visualization of rewards.

### 2. 🧪 Experiments & MLOps Integration
- **MLflow Tracking**: Every experiment run is automatically tracked. This includes logging of configurations, hyperparameters, metrics, and model artifacts (including environments and YAML configurations for RL runs).
- **Job Manager**: Comprehensive dashboard for background job control. Train complex models and RL agents in the background via subprocesses without locking the Streamlit UI.
- **Data Lake & Trajectories**: Save datasets, images, and RL agent trajectories (states, actions, rewards, terminal signals) to a centralized Data Lake.

### 3. 🚀 Serving & Deployment
- **FastAPI Serving**: Production-ready API (`api.py`) for real-time inference with API Key security.
- **Live Telemetry & Data Drift**: Input data and predictions are logged dynamically for drift and performance analysis.
- **Playground**: Interactive UI to test registered models via JSON or CSV Batch.

---

## 📋 Supported Task Types

| Modality | Task Type | Brief Description | Main Metrics |
|---|---|---|---|
| **Tabular** | `classification` | Predict a discrete class label. | `accuracy`, `f1`, `precision`, `recall`, `roc_auc` |
| **Tabular** | `regression` | Predict a continuous numeric target. | `r2`, `rmse`, `mae` |
| **Tabular** | `forecast` | Predict future values using historical temporal data. | `r2`, `rmse`, `mae` |
| **Tabular** | `clustering` | Group samples by similarity without labels. | `silhouette` |
| **Tabular** | `anomaly_detection` | Detect outliers or rare abnormal patterns. | `f1`, anomaly ratio |
| **Tabular** | `ranking` | Score items for ordered relevance. | `ndcg` |
| **Tabular** | `multi_label` | Predict multiple labels per row (multi-target). | `f1_micro`, `subset_accuracy` |
| **Tabular** | `multi_task` | Predict multiple disparate classification targets concurrently. | `f1_micro`, `subset_accuracy` |
| **Tabular** | `association_rules` | Discover co-occurrence rules. | `rule_score`, `lift` |
| **Computer Vision** | `image_classification` | Assign one class to each image. | `val_acc`, `val_loss` |
| **Computer Vision** | `image_segmentation` | Pixel-wise semantic segmentation. | `val_score`, `val_loss` |
| **Computer Vision** | `object_detection` | Detect objects and bounding boxes. | Benchmark metrics |
| **Computer Vision** | `pose_estimation` | Estimate keypoints/body joints. | Keypoint accuracy |
| **Reinforcement Learning** | `rl_agent` | Train an agent to maximize reward. | `episode_reward`, `mean_reward` |

---

## 🚀 Advanced ML Capabilities (New Features)

## 🆕 What's New (Recent)

- **Advanced ML Preprocessing & Modeling Pipeline**:
  - **Temporal & Text Characteristics**: Tabular datasets now support specifying "Contains Temporal Data" (automatically applies chronological validation splits and generates lags/rolling window features) and "Contains Text / NLP Data" (automatically vectorizes text columns using a high-performance TF-IDF pipeline).
  - **Forecast Task Type**: Replaces the old hardcoded Time Series task with a dedicated Forecast engine integrated across all frameworks.
  - **Multi-Task Classification**: Support for predicting multiple target columns concurrently. The interface automatically orchestrates separate training runs for each target if the framework does not support it natively.
  - **Semi-Supervised Learning**: Support for Self-Training Classification using target columns with unlabeled samples (marked as `-1` or `NaN`). The training pipeline dynamically wraps base classifiers in a `SelfTrainingClassifier` wrapper.

---

## 📂 Project Structure

- `app.py`: Main Streamlit UI with unified mode switching (Tabular, CV, RL) and optimized caching.
- `api.py`: FastAPI implementation for model serving and telemetry.
- `src/engines/`: Contains the core machine learning logic.
  - `reinforcement_learning.py`: RL logic handling Online/Offline training and Callbacks.
- `src/tracking/`: Subprocess Job Manager and MLflow tracking wrappers.
- `src/core/`: Data processors, Drift detection logic.
- `src/ui/`: Design system and custom CSS components.
- `Dockerfile` & `docker-compose.yml`: Infrastructure containerization.
- `requirements.txt`: Project dependencies with pinned environment libraries.

---

## 🚀 How to Run

### 🐳 Via Docker (Recommended)
AutoMLOps Studio is fully dockerized to ensure it can be easily deployed and used in any environment.

```bash
docker-compose up --build
```
This single command spins up three services:
- **Dashboard (Streamlit)**: `http://localhost:8501`
- **Serving API (FastAPI)**: `http://localhost:8000`
- **MLflow UI**: `http://localhost:5000`

### 🐍 Locally (Python)
Ensure you have Python installed.

1. **Install Requirements**:
```bash
pip install -r requirements.txt
```
*(Dependencies like `stable-baselines3[extra]`, `gymnasium`, and `d3rlpy` are included for full ML and RL support).*

2. **Run the Streamlit Dashboard**:
```bash
python -m streamlit run app.py
```

3. **Run the Serving API (Optional)**:
```bash
uvicorn api:app --reload --port 8000
```

4. **Run the MLflow UI (Optional)**:
```bash
mlflow ui --port 5000
```

---

**Developed by Pedro Morato Lahoz.**
