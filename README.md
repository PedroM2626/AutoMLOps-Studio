# 🚀 AutoMLOps Studio

### Enterprise-grade Automated Machine Learning & MLOps Platform

[![Version](https://img.shields.io/badge/Version-v2.0.0-green)](https://github.com/PedroM2626/automlops-studio)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PedroM2626/AutoMLOps-Studio)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Integrated-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

**AutoMLOps Studio** is an "end-to-end" educational platform designed to simplify the Machine Learning lifecycle. Developed **by a student, for students**, the project provides an intuitive interface to explore everything from data ingestion to production model monitoring.

**🔗 Access the live Demo:** [Hugging Face Spaces - AutoMLOps Studio](https://huggingface.co/spaces/PedroM2626/AutoMLOps-Studio)

---

## 🎯 Objective & Problem Statement

Learning MLOps often requires dealing with complex infrastructures before even understanding the core concepts. This project solves that by centralizing:
- **Unified Workflow**: A clear journey from data upload to deployment.
- **Visual Experimentation**: Visualize the impact of hyperparameters and architectures in real-time.
- **Production Concepts**: Learn about Data Drift, Model Registry, and Performance Monitoring without the need to configure complex servers.

## 👥 Target Audience

- **Data Science Students**: Looking to consolidate theoretical knowledge with visual practice.
- **ML Enthusiasts**: Who need a fast tool to prototype models and test hypotheses.
- **Junior MLOps Developers**: Who want to understand the integration between tools like MLflow, Optuna, and prediction APIs.

---

## ✨ Features & Technical Details

### 1. 📊 Data Management & Drift Analysis
- **Smart Ingestion**: CSV upload with versioned storage in the local Data Lake.
- **Integrated Drift Detection**: Statistical analysis (KS Test) and **Deepchecks** integration for automatic reports.

### 2. 🤖 Training Configuration & NLP
- **Multi-Task Support**: Classification, Regression, Clustering, and Time Series.
- **NLP & Transformers**: Support for TF-IDF and Hugging Face Transformers.
- **Advanced Optimization**: Optuna with Bayesian Optimization.

### 3. 🧪 Experiments System (New v2.0!)
- **Concurrent Training**: Submit multiple training sessions simultaneously (Big Tech style).
- **Background Execution**: Training no longer blocks the main UI. Each run executes in an isolated subprocess.
- **Job Management Dashboard**: Pause, Resume, Cancel, or Delete jobs in real-time.
- **360° Detail Panel**: 
    - **Progress**: Live optimization charts.
    - **Logs**: Real-time log streaming from the backend.
    - **Results**: Complete reports and performance charts.
    - **MLflow**: Deep integration with parameters, metrics, and artifacts.

### 4. 🚀 Performance & UX (New!)
- **Fragment Refresh (`st.fragment`)**: Smart updates restricted to the experiments area, keeping the rest of the app stable and responsive.
- **API Caching**: Drastic reduction in latency when viewing MLflow runs.

### 5. ⚖️ Monitoring & Stability
- **Robustness Tests**: Stability evaluation against Seeds, Splits, and Hyperparameter variation.
- **Production Monitoring**: Comparison between training and production data.

### 6. 📦 Model Registry & Deployment
- **Registry via MLflow**: Version control and stages (Staging/Production).
- **Inference Test**: Playground to test registered models with CSV/JSON inputs.

---

## 📂 Project Structure

- `app.py`: Main Streamlit interface with Fragment support.
- `training_manager.py`: (New!) Concurrent job and subprocess manager.
- `automl_engine.py`: Training and optimization engine.
- `mlops_utils.py`: MLflow integration and metadata caching.
- `stability_engine.py`: Robustness testing engine.

---

## 🚀 How to Run

### 🐳 Via Docker (Recommended)
```bash
docker-compose up --build
```

### 🐍 Locally (Python)
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

---

**Developed by Pedro Morato Lahoz.**
