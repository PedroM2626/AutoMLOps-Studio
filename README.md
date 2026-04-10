# 🚀 AutoMLOps Studio

### Comprehensive Automated Machine Learning & MLOps Platform

[![Version](https://img.shields.io/badge/Version-v5.0.1-blue)](https://github.com/PedroM2626/automlops-studio)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PedroM2626/AutoMLOps-Studio)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Integrated-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

**AutoMLOps Studio** is an "end-to-end" educational platform designed to simplify the Machine Learning lifecycle. Developed **by a student, for students**, the project provides an intuitive interface to explore everything from data ingestion to production model monitoring.

**🔗 Access the live Demo:** [Streamlit Cloud - AutoMLOps Studio](https://automlops-studio.streamlit.app/)

---

## 🌟 What's New in v5.0.0

### 🤖 Unified AutoML Studio
- **Single Unified Entry Point**: AutoML now starts in one workspace with a modality selector (**Tabular** or **Computer Vision**)
- **Conditional Options**: Dynamic interface adapts to selected modality and task type
- **Simplified Navigation**: No duplicated training flows between separate tabs

### 🖼️ Enhanced Computer Vision
- **New Vision Tasks**: Added **Anomaly Detection** and **Pose Estimation**
- **Task-Specific Configuration**: Different options for classification, multi-label, segmentation, detection, anomaly detection, and pose estimation
- **Dynamic UI**: Options appear/hide based on selected CV task type
- **Integrated Pipeline**: CV models now follow the same MLOps pipeline as tabular models

### 🎯 Smart Task Selection
- **New Tabular Tasks**: Added **Association Rules**, **Ranking**, and **Multi-Label** for tabular datasets
- **Unified Task Types**: Tabular and vision tasks are available from the same AutoML entry point
- **Context-Aware Options**: Parameters change dynamically based on data and task type
- **Mixed Data Support**: Handle both tabular and image data in unified workflows

### ✅ Pipeline Coverage Improvements
- **Tabular Multi-Label End-to-End**: Multi-target column selection and processing now supported in the training pipeline
- **Association Rules Engine**: Built-in rule-mining flow with support, confidence, and lift-based scoring
- **Ranking Metrics**: Ranking workflows now include ranking-aware optimization metrics (for example, NDCG)

## 📋 Supported Task Types (Tabular vs CV)

| Modality | Task Type | Brief Description | Main Metrics |
|---|---|---|---|
| Tabular | `classification` | Predict a discrete class label. | `accuracy`, `f1`, `precision`, `recall`, `roc_auc` |
| Tabular | `regression` | Predict a continuous numeric target. | `r2`, `rmse`, `mae` |
| Tabular | `time_series` | Forecast future values from temporal data. | `rmse`, `mae`, `mape` |
| Tabular | `clustering` | Group samples by similarity without labels. | `silhouette` |
| Tabular | `anomaly_detection` | Detect outliers or rare abnormal patterns. | `f1` (when labels exist), anomaly ratio/count |
| Tabular | `dimensionality_reduction` | Reduce feature space while preserving signal. | `explained_variance` |
| Tabular | `ranking` | Score items for ordered relevance. | `ndcg`, `rmse`, `mae` |
| Tabular | `multi_label` | Predict multiple labels per row (multi-target). | `f1_micro`, `subset_accuracy`, `precision_micro`, `recall_micro`, `hamming_loss` |
| Tabular | `association_rules` | Discover co-occurrence rules in tabular/binary patterns. | `rule_score`, `rule_count`, `avg_lift` |
| Computer Vision | `image_classification` | Assign one class to each image. | `val_acc`, `val_loss` |
| Computer Vision | `image_multi_label` | Assign multiple labels to each image. | `val_acc` (exact match), `val_loss` |
| Computer Vision | `image_segmentation` | Pixel-wise semantic segmentation. | `val_score`, `val_loss` |
| Computer Vision | `object_detection` | Detect objects and bounding boxes. | Baseline loop enabled; custom detector metrics can be added per dataset |
| Computer Vision | `image_anomaly_detection` | Classify images as normal vs anomalous patterns. | `val_acc`, `val_loss` |
| Computer Vision | `pose_estimation` | Estimate keypoints/body joints from images. | Baseline loop enabled; keypoint metrics depend on annotation format |

> Notes:
> - Tabular metrics are configurable in the AutoML optimization step.
> - Some CV tasks (detection/pose) are scaffolded and ready in pipeline/UI; task-specific benchmark metrics (for example mAP/OKS) can be plugged in according to annotation standards.

### 🛠️ Enhanced Training Presets
- **Fast Preset Fix**: Resolved premature trial termination issue, ensuring consistent behavior across all presets.
- **Customizable Trials**: Added flexibility for manual configuration of trials and timeouts in "Custom" mode.

### 🎯 Improved Model Selection
- **`bagging` Removed**: Hidden from user-facing model lists to streamline the selection process.
- **Dynamic Forms**: Replaced manual JSON editor with interactive forms for algorithm parameter tuning.

### ⚡ MLflow Integration
- **Warning Fixes**: Addressed `os` variable scope conflicts for seamless logging.
- **Enhanced Tracking**: Improved parameter and metric logging for better experiment reproducibility.

### 🖥️ Streamlit Wizard Updates
- **Robustness Improvements**: Enhanced error handling for missing dataset versions and invalid configurations.
- **UI Refinements**: Polished interface for a smoother user experience.

---

## 🎯 Objective & Problem Statement

Learning MLOps often requires dealing with complex infrastructures before even understanding the core concepts. This project solves that by centralizing:
- **Unified Workflow**: A clear journey from data upload to deployment.
- **Visual Experimentation**: Visualize the impact of hyperparameters and architectures in real-time.
- **Production Concepts**: Learn about Data Drift, Model Serving, and Performance Monitoring without the need to configure complex servers.


### 3. 🧪 Experiments & MLOps
- **Job Manager**: Comprehensive dashboard for background job control.
- **MLflow Tracking**: Integrated logging of params, metrics, and architecture diagrams.
- **Reports**: Automated performance reports (ROC, PR, Residuals, SHAP) saved as artifacts.
- **Optimized Performance**: Streamlit caching system for fast data loading and real-time updates.

### 4. 🚀 Serving & Deployment
- **FastAPI Serving**: Production-ready API for real-time inference with API Key security.
- **Live Telemetry**: Input data and predictions are logged for drift and performance analysis.
- **Playground**: Interactive UI to test registered models via JSON or CSV Batch.

---

## 📂 Project Structure

- `app.py`: Main Streamlit UI with unified AutoML mode switching (Tabular/CV) and optimized caching.
- `src/core/`: Data processor, Drift detection, and Trainer constants.
- `src/engines/`: Classical (Scikit-Learn/XGB), Vision (Torch), and Stability engines with extended task-type support.
- `src/tracking/`: Subprocess Job Manager and MLflow wrappers.
- `src/ui/`: Design system and custom CSS components.
- `api.py`: FastAPI implementation for model serving.

---

## 🚀 How to Run

### 🐳 Via Docker (Fastest)
```bash
docker-compose up --build
```
This starts the **Dashboard (8501)**, the **Serving API (8000)**, and the **MLflow UI (5000)**.

### 🐍 Locally (Python)
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## ❗ Discontinued: Reflex Interface

The Reflex-based interface was **discontinued and completely removed** from the project in version v4.8.0. 

### 📋 Reasons for Discontinuation:
1. **Performance & Complexity**: The Streamlit interface, after implementing comprehensive caching strategies (`st.cache_data`), achieved superior performance and responsiveness.
2. **Development Velocity**: Streamlit's data-first approach allows for faster iteration and implementation of MLOps features compared to the overhead of managing state in Reflex.
3. **Redundancy**: Maintaining two separate UIs (Streamlit + Reflex) created unnecessary complexity without significant user benefits.
4. **Ecosystem Maturity**: Streamlit's mature ecosystem for data visualization (Plotly, Matplotlib, SHAP) and MLflow integration proved more reliable for MLOps workflows.

### ✅ Current State:
- **Streamlit Only**: The project now focuses exclusively on a single, highly-optimized Streamlit interface.
- **Enhanced Performance**: Implemented intelligent caching for Data Lake operations, MLflow queries, and DataFrame loading.
- **Unified Experience**: AutoML now uses one workflow entry point with modality-based dynamic options.
- **Broader Task Coverage**: New task types for both tabular and vision are integrated into the same MLOps pipeline.

---

**Developed by Pedro Morato Lahoz.**
