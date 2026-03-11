# 🚀 AutoMLOps Studio

### Comprehensive Automated Machine Learning & MLOps Platform

[![Version](https://img.shields.io/badge/Version-v4.5.2-blue)](https://github.com/PedroM2626/automlops-studio)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PedroM2626/AutoMLOps-Studio)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Integrated-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

**AutoMLOps Studio** is an "end-to-end" educational platform designed to simplify the Machine Learning lifecycle. Developed **by a student, for students**, the project provides an intuitive interface to explore everything from data ingestion to production model monitoring.

**🔗 Access the live Demo:** [Hugging Face Spaces - AutoMLOps Studio](https://huggingface.co/spaces/PedroM2626/AutoMLOps-Studio)

---

## 🌟 What's New in v4.5.0

### 🏗️ Integrated Custom Ensemble Builder
The **Custom Ensemble Builder** is now fully integrated into the **Manual (Select)** mode. Users can hand-pick base models and then add any number of **Voting, Stacking, or Bagging** configurations using those models. This merge provides a unified experience for building sophisticated model combinations.

### 🤖 Smart Automatic Configuration
An improved **Automatic (Preset)** mode now handles ensemble creation intelligently. Based on your **Training Focus** (Single, Ensembles, or Both) and **Deep Learning** settings, the engine automatically selects the best candidate models and configures optimized ensembles without requiring manual input.

### ⚡ Background Infrastructure (Big Tech Style)
The entire training pipeline now runs in **isolated background subprocesses**. This prevents UI freezes and allows for:
- **Live Log Streaming**: Watch the model training logs in real-time.
- **Concurrent Experiments**: Queue and run multiple trials simultaneously.
- **Job Control**: Pause, Resume, or Cancel training jobs without losing the main application state.

### 🎯 Refined Model Privacy
Internal training methods like `custom_voting`, `bagging`, and `stacking_ensemble` are now hidden from the primary model selection list. The UI focuses purely on **Base Estimators**, ensuring that ensembles are built explicitly through the builder or automatically by the preset engine.

---

## 🎯 Objective & Problem Statement

Learning MLOps often requires dealing with complex infrastructures before even understanding the core concepts. This project solves that by centralizing:
- **Unified Workflow**: A clear journey from data upload to deployment.
- **Visual Experimentation**: Visualize the impact of hyperparameters and architectures in real-time.
- **Production Concepts**: Learn about Data Drift, Model Serving, and Performance Monitoring without the need to configure complex servers.

## ✨ Features & Technical Details

### 1. 📊 Data Engineering & Drift
- **Smart Data Lake**: Versioned ingestion for CSV, JSON, and Parquet.
- **Schema Control**: Manual data type overrides and interactive column selection.
- **Integrated Monitoring**: Drift detection (KS-Test) comparing current data vs. training history.

### 2. 🤖 AutoML Engine (v4.0+)
- **Multi-Task**: Classification, Regression, Clustering, and Time Series.
- **NLP & Transformers**: TF-IDF and Hugging Face integration for text data.
- **Computer Vision**: PyTorch-based training with MLflow tracking for vision tasks.
- **Optimization**: Optuna with Bayesian Search and early stopping.

### 3. 🧪 Experiments & MLOps
- **Job Manager**: Comprehensive dashboard for background job control.
- **MLflow Tracking**: Integrated logging of params, metrics, and architecture diagrams.
- **Reports**: Automated performance reports (ROC, PR, Residuals, SHAP) saved as artifacts.

### 4. 🚀 Serving & Deployment
- **FastAPI Serving**: Production-ready API for real-time inference with API Key security.
- **Live Telemetry**: Input data and predictions are logged for drift and performance analysis.
- **Playground**: Interactive UI to test registered models via JSON or CSV Batch.

---

## 📂 Project Structure

- `app.py`: Main Streamlit UI with state-of-the-art design and fragment support.
- `src/core/`: Data processor, Drift detection, and Trainer constants.
- `src/engines/`: Classical (Scikit-Learn/XGB), Vision (Torch), and Stability engines.
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

**Developed by Pedro Morato Lahoz.**
