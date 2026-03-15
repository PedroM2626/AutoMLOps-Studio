# 🚀 AutoMLOps Studio

### Comprehensive Automated Machine Learning & MLOps Platform

[![Version](https://img.shields.io/badge/Version-v4.7.1-blue)](https://github.com/PedroM2626/automlops-studio)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PedroM2626/AutoMLOps-Studio)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Integrated-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

**AutoMLOps Studio** is an "end-to-end" educational platform designed to simplify the Machine Learning lifecycle. Developed **by a student, for students**, the project provides an intuitive interface to explore everything from data ingestion to production model monitoring.

**🔗 Access the live Demo:** [Hugging Face Spaces - AutoMLOps Studio](https://huggingface.co/spaces/PedroM2626/AutoMLOps-Studio)

---

## 🌟 What's New in v4.6.0

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

### Reflex Interface
```bash
pip install -r requirements.txt
reflex run
```
The Reflex app adds a modular native control plane in dark mode, covering data operations, AutoML jobs, experiments, registry/deploy flows, DagsHub tracking integration, monitoring, and computer vision workflows without depending on the Streamlit interface.

---

**Developed by Pedro Morato Lahoz.**
