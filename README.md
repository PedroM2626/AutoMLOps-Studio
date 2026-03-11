# 🚀 AutoMLOps Studio

### Comprehensive Automated Machine Learning & MLOps Platform

[![Version](https://img.shields.io/badge/Version-v4.5.0-blue)](https://github.com/PedroM2626/automlops-studio)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PedroM2626/AutoMLOps-Studio)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Integrated-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

**AutoMLOps Studio** is an "end-to-end" educational platform designed to simplify the Machine Learning lifecycle. Developed **by a student, for students**, the project provides an intuitive interface to explore everything from data ingestion to production model monitoring.

**🔗 Access the live Demo:** [Hugging Face Spaces - AutoMLOps Studio](https://huggingface.co/spaces/PedroM2626/AutoMLOps-Studio)

---

## 🌟 What's New in v4.4.0

### 🛠️ Custom Ensemble Builder & Advanced Selection
A sophisticated **Custom Ensemble Builder** has been introduced, allowing users to choose between **Voting** and **Stacking** strategies. You can now manually select base estimators, meta-models, and configure voting types (Soft/Hard) directly from the UI. The model selection logic has been refined to strictly respect manual user choices, overriding default filters.

### ⚡ Optimized Training Pipeline
Ensemble models and static models are now automatically detected and optimized to run a **single trial**. This eliminates redundant hyperparameter searches for composite models, significantly speeding up the training process while maintaining high performance.

### 📝 Descriptive Experiment Naming
Experiments and charts now feature **Detailed Descriptive Names**. For example, a custom ensemble will appear as *"Custom Voting Ensemble (Random Forest, XGBoost)"*, providing immediate clarity in MLflow and the UI dashboard.

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
- **Smart Ingestion**: CSV, JSON, Parquet, and TXT upload with versioned storage in the local Data Lake.
- **Manual Dataset Parsing**: Control over delimiters, encoding, and file formats during upload.
- **Interactive Schema Editor**: Inclusion/Exclusion of columns and manual data type overrides before training.
- **Integrated Drift Detection**: Statistical analysis (KS Test) and **Deepchecks** integration for automatic reports.

### 2. 🤖 Training Configuration, NLP & Vision
- **Multi-Task Support**: Classification, Regression, Clustering, Time Series, and **Computer Vision** (Classification, Segmentation, Multi-label).
- **Custom Ensemble Builder**: Dedicated UI to build **Voting, Stacking, and Bagging** models. Select your own base learners and meta-estimators.
- **Precision Model Selection**: Toggle between Automatic, Manual, and Custom modes. Explicit manual selections always bypass general filters (Deep Learning/Ensemble modes).
- **NLP & Transformers**: Support for TF-IDF and Hugging Face Transformers. Full English support for embeddings and tokenization.
- **Computer Vision**: Integrated training with MLflow tracking, data augmentation config, and automated consumption code generation.
- **Advanced Optimization**: Optuna with Bayesian Optimization.
- **Advanced Splitting Strategies**: Random, Chronological (time-based), and Manual (col-based) splits with visual representation.

### 3. 🧪 Experiments System
- **Concurrent Training**: Submit multiple training sessions simultaneously (Big Tech style).
- **Background Execution**: Training no longer blocks the main UI. Each run executes in an isolated subprocess.
- **Job Management Dashboard**: Pause, Resume, Cancel, or Delete jobs in real-time.
- **360° Detail Panel**: 
    - **Progress**: Live optimization charts.
    - **Logs**: Real-time log streaming from the backend.
    - **Results**: Complete reports, performance charts, and SHAP summaries.
    - **MLflow**: Deep integration with parameters, metrics, and artifacts.

### 4. 🚀 Performance & UX
- **Fragment Refresh (`st.fragment`)**: Smart updates restricted to the experiments area, keeping the rest of the app stable and responsive.
- **API Caching**: Drastic reduction in latency when viewing MLflow runs.

### 5. ⚖️ Monitoring & Stability
- **Robustness Tests**: Stability evaluation against Seeds, Splits, and Hyperparameter variation.
- **Production Monitoring**: Comparison between training and production data.

### 6. 📦 Model Registry & Deployment
- **Registry via MLflow**: Version control and stages (Staging/Production).
- **Inference Test & Playground**: 
    - **Tabular & NLP**: Real-time testing via JSON payload or **Batch Inference** via CSV upload with automatic results download.
    - **Vision**: Live playground to upload images and see predictions/masks in real-time.

---

## 📂 Project Structure

- `app.py`: Main Streamlit interface with Fragment support and professional UI styling.
- `src/core/`: Data Lake storage, Drift detection, and Preprocessing pipelines.
- `src/engines/`: Classical ML (Tabular), Computer Vision (PyTorch/Transformers), and Stability engines.
- `src/tracking/`: Background Job Manager and MLflow tracking wrappers.
- `src/utils/`: SHAP Explainers, Helper functions, and Automated Code Generators.
- `src/ui/`: Custom CSS styling and design system tokens.

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
