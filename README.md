# 🚀 AutoMLOps Studio

### Comprehensive Automated Machine Learning & MLOps Platform

[![Version](https://img.shields.io/badge/Version-v4.9.1-blue)](https://github.com/PedroM2626/automlops-studio)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PedroM2626/AutoMLOps-Studio)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Integrated-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

**AutoMLOps Studio** is an "end-to-end" educational platform designed to simplify the Machine Learning lifecycle. Developed **by a student, for students**, the project provides an intuitive interface to explore everything from data ingestion to production model monitoring.

**🔗 Access the live Demo:** [Streamlit Cloud - AutoMLOps Studio](https://automlops-studio.streamlit.app/)

---

## 🌟 What's New in v4.9.1

### 🤖 Unified AutoML Studio
- **Computer Vision Integration**: CV tasks now integrated within the main AutoML workflow
- **Conditional Options**: Dynamic interface that adapts based on selected task type
- **Three Workflow Modes**: Classical ML, Computer Vision, and Unified AutoML approaches

### 🖼️ Enhanced Computer Vision
- **Task-Specific Configuration**: Different options for classification, segmentation, and detection
- **Dynamic UI**: Options appear/hide based on selected CV task type
- **Integrated Pipeline**: CV models now follow the same MLOps pipeline as tabular models

### 🎯 Smart Task Selection
- **Unified Task Types**: All ML tasks (tabular and vision) in one interface
- **Context-Aware Options**: Parameters change dynamically based on data and task type
- **Mixed Data Support**: Handle both tabular and image data in unified workflows

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

- `app.py`: Main Streamlit UI with state-of-the-art design and optimized caching.
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
- **Unified Experience**: All Computer Vision and AutoML features are now fully integrated into the Streamlit workflow.

---

**Developed by Pedro Morato Lahoz.**
