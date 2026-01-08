#!/usr/bin/env python3
"""
🎯 MLOPS ENTERPRISE - UNIVERSAL FRAMEWORK
Recursos:
- Rastreamento completo no DagsHub/MLflow para todos os módulos.
- ML Clássico, Transformers, Computer Vision, Time Series e Clustering.
- Otimização (Optuna), Monitoramento (Evidently) e Serving (FastAPI).
"""

import os
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# MLOps, Tracking & Optimization
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.transformers
import dagshub
import optuna

# Deep Learning & Stats
import torch
try:
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, 
                              TrainingArguments, Trainer, DataCollatorWithPadding)
    from datasets import Dataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False

# ML Clássico e Utilidades
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

class MLOpsEnterprise:
    def __init__(self, repo_owner='PedroM2626', repo_name='free-mlops'):
        load_dotenv()
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_dagshub()

    def _setup_dagshub(self):
        try:
            dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
            print("✅ Conectado ao DagsHub/MLflow")
        except Exception as e:
            print(f"⚠️ Erro DagsHub: {e}")

    def _log_metrics_and_plots(self, metrics, artifacts=None):
        """Helper para logar métricas e artefatos de forma consistente."""
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        if artifacts:
            for path in artifacts:
                mlflow.log_artifact(path)

    # --- MÓDULO 1: ML CLÁSSICO (AGNOSTICO) ---
    def train_classic_ml(self, task='classification', data_path='processed_train.csv'):
        print(f"\n� Treinando ML Clássico ({task})...")
        df = pd.read_csv(data_path)
        
        # Detecção automática: NLP ou Tabular
        is_nlp = 'text_lemmatized' in df.columns
        target = 'sentiment' if 'sentiment' in df.columns else df.columns[-1]
        
        X = df['text_lemmatized'].fillna('') if is_nlp else df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        mlflow.set_experiment(f"/classic_{task}")
        with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            if is_nlp:
                model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier())])
            else:
                model = RandomForestClassifier() if task == 'classification' else RandomForestRegressor()
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            metric_val = accuracy_score(y_test, preds) if task == 'classification' else mean_squared_error(y_test, preds)
            metric_name = "accuracy" if task == 'classification' else "mse"
            
            self._log_metrics_and_plots({metric_name: metric_val})
            mlflow.sklearn.log_model(model, "model", registered_model_name=f"classic_{task}_model")
            print(f"✅ {metric_name.capitalize()}: {metric_val:.4f}")

    # --- MÓDULO 2: TIME SERIES (PROPHET) ---
    def train_time_series(self, data_path=None):
        if not HAS_PROPHET:
            print("⚠️ Prophet não instalado."); return
        
        print("\n� Treinando Série Temporal (Prophet)...")
        # Dados sintéticos se não houver path
        if data_path is None:
            df = pd.DataFrame({'ds': pd.date_range('2023-01-01', periods=100), 'y': np.random.randn(100).cumsum()})
        else:
            df = pd.read_csv(data_path)

        mlflow.set_experiment("/time_series")
        with mlflow.start_run(run_name="prophet_run"):
            model = Prophet()
            model.fit(df)
            
            # Log de "métricas" (aqui simplificado)
            mlflow.log_param("periods", len(df))
            mlflow.sklearn.log_model(model, "model", registered_model_name="ts_prophet_model")
            print("✅ Modelo Prophet registrado.")

    # --- MÓDULO 3: CLUSTERING (K-MEANS) ---
    def train_clustering(self, n_clusters=3, data_path='processed_train.csv'):
        print(f"\n🧬 Treinando Agrupamento (K-Means, k={n_clusters})...")
        df = pd.read_csv(data_path)
        X = df.select_dtypes(include=[np.number]).fillna(0)
        if X.empty: X = np.random.rand(100, 2) # Fallback

        mlflow.set_experiment("/clustering")
        with mlflow.start_run(run_name="kmeans_run"):
            model = KMeans(n_clusters=n_clusters)
            model.fit(X)
            score = silhouette_score(X, model.labels_)
            
            # Plot PCA para visualização
            pca = PCA(2).fit_transform(X)
            plt.figure(figsize=(8,6))
            plt.scatter(pca[:,0], pca[:,1], c=model.labels_)
            plt.savefig("cluster_plot.png")
            
            self._log_metrics_and_plots({"silhouette_score": score}, ["cluster_plot.png"])
            mlflow.sklearn.log_model(model, "model", registered_model_name="clustering_model")
            print(f"✅ Silhouette Score: {score:.4f}")

    # --- MÓDULO 4: COMPUTER VISION (YOLOv8) ---
    def train_cv(self, task='detect', data_config=None, model_type='yolov8n.pt', epochs=5):
        """
        Módulo Universal de CV: Classificação, Detecção ou Segmentação.
        task: 'classify', 'detect', 'segment'
        """
        if not HAS_ULTRALYTICS:
            print("⚠️ Ultralytics não instalado. Execute: pip install ultralytics")
            return

        print(f"\n🖼️ Treinando Computer Vision (YOLOv8 - {task})...")
        mlflow.set_experiment(f"/cv_{task}")

        # Iniciar run do MLflow
        with mlflow.start_run(run_name=f"yolo_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Carregar modelo pré-treinado
            model = YOLO(model_type)

            # Fine-tuning se houver config de dados, caso contrário apenas logamos o modelo base
            if data_config:
                results = model.train(data=data_config, epochs=epochs, imgsz=640)
                # Logar métricas de treino
                mlflow.log_metrics(results.results_dict)
            
            # Logar o modelo no MLflow
            # Nota: YOLO não tem log nativo perfeito no MLflow v2.x sem callback, 
            # então logamos o arquivo .pt como artefato e o modelo via wrapper se necessário
            model_path = f"best_{task}.pt"
            model.export(format='onnx') # Exemplo de exportação
            mlflow.log_artifact(model_type)
            
            print(f"✅ Modelo YOLO ({task}) registrado no DagsHub.")
            return model

    # --- MÓDULO 5: MONITORAMENTO & API ---
    def detect_drift(self, reference_df, current_df):
        if not HAS_EVIDENTLY: return
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        report.save_html("drift_report.html")
        mlflow.log_artifact("drift_report.html")

    def generate_serving_api(self, model_name):
        print(f"\n🚀 Gerando API de Serving para {model_name}...")
        api_code = f"""
import uvicorn
from fastapi import FastAPI
import mlflow.sklearn
import dagshub
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()
dagshub.init(repo_owner='{self.repo_owner}', repo_name='{self.repo_name}', mlflow=True)

app = FastAPI(title="MLOps Enterprise API")

# Tentar carregar a versão mais recente do modelo registrado
try:
    model_uri = f"models:/{model_name}/latest"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Modelo '{{model_name}}' carregado com sucesso!")
except Exception as e:
    print(f"⚠️ Erro ao carregar 'latest', tentando versão 1: {{e}}")
    model_uri = f"models:/{model_name}/1"
    model = mlflow.sklearn.load_model(model_uri)

class InputData(BaseModel):
    text: str

@app.post("/predict")
def predict(item: InputData):
    # Ajuste dinâmico dependendo do tipo de entrada que o modelo espera
    prediction = model.predict([item.text])[0]
    return {{"prediction": str(prediction)}}

if __name__ == "__main__":
    # Importante: host 0.0.0.0 para funcionar dentro do Docker
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        with open("app_serving.py", "w", encoding="utf-8") as f:
            f.write(api_code)
        print("✅ API gerada com sucesso em 'app_serving.py'.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['classic', 'ts', 'cluster', 'all'], default='all')
    args = parser.parse_args()
    
    m = MLOpsEnterprise()
    
    if args.task in ['classic', 'all']:
        m.train_classic_ml(task='classification')
    if args.task in ['ts', 'all']:
        m.train_time_series()
    if args.task in ['cluster', 'all']:
        m.train_clustering()

if __name__ == "__main__":
    main()
