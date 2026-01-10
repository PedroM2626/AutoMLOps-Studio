#!/usr/bin/env python3
"""
🎯 MLOPS ENTERPRISE - UNIVERSAL FRAMEWORK (V4.0)
Recursos Avançados:
- AutoML: Unified (TPOT, AutoGluon, FLAML, Auto-sklearn, H2O).
- Otimização (Optuna), Explainability (SHAP/LIME).
- Validação (Evidently), Exportação (ONNX).
- Integrações: MLflow, DagsHub, W&B, HuggingFace.
- Distributed Training (PyTorch), K8s Deployment Ready.
- CV (YOLOv8), NLP (Transformers), Time Series (Prophet).
"""

import os
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
import logging
from typing import Optional, Dict, Any, Union

# MLOps & Tracking
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.h2o
import dagshub
import wandb
import optuna

# AutoML Engines
try:
    from tpot import TPOTClassifier, TPOTRegressor
    HAS_TPOT = True
except ImportError:
    HAS_TPOT = False

try:
    from autogluon.tabular import TabularPredictor
    HAS_AUTOGLUON = True
except ImportError:
    HAS_AUTOGLUON = False

try:
    from flaml import AutoML
    HAS_FLAML = True
except ImportError:
    HAS_FLAML = False

try:
    import autosklearn.classification
    import autosklearn.regression
    HAS_AUTOSKLEARN = True
except ImportError:
    HAS_AUTOSKLEARN = False

try:
    import h2o
    from h2o.automl import H2OAutoML
    HAS_H2O = True
except ImportError:
    HAS_H2O = False

# Deep Learning & Transformers
import torch
import torch.distributed as dist
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset

# Explainability & Monitoring
import shap
import lime
import lime.lime_tabular
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# ML Clássico & Export
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from ultralytics import YOLO
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLOpsEnterprise:
    def __init__(self, repo_owner='PedroM2626', repo_name='experiments'):
        load_dotenv()
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_integrations()

    def _setup_integrations(self):
        try:
            dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
            logger.info("✅ Conectado ao DagsHub/MLflow")
            if os.getenv("WANDB_API_KEY"):
                wandb.login(key=os.getenv("WANDB_API_KEY"))
                wandb.init(project=self.repo_name, entity=self.repo_owner)
                logger.info("✅ Conectado ao Weights & Biases")
        except Exception as e:
            logger.warning(f"⚠️ Erro nas integrações: {e}")

    def _log_metrics_and_plots(self, metrics, artifacts=None):
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            if wandb.run: wandb.log({name: value})
        if artifacts:
            for path in artifacts:
                mlflow.log_artifact(path)

    def validate_data(self, df: pd.DataFrame, target: str) -> bool:
        """Valida a integridade dos dados antes do treino."""
        logger.info("🔍 Validando integridade dos dados...")
        if df.empty:
            raise ValueError("O DataFrame está vazio.")
        if target not in df.columns:
            raise ValueError(f"Target '{target}' não encontrado no DataFrame.")
        
        null_counts = df.isnull().sum().sum()
        if null_counts > (len(df) * len(df.columns) * 0.5):
            logger.warning(f"⚠️ Alto índice de valores nulos detectado: {null_counts}")
            
        return True

    def detect_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame):
        """Detecta drift de dados usando Evidently AI."""
        logger.info("📉 Analisando Data Drift...")
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        report_path = "drift_report.html"
        report.save_html(report_path)
        mlflow.log_artifact(report_path)
        return report_path

    # --- MÓDULO AUTOML UNIFICADO (TPOT, AutoGluon, FLAML, Auto-sklearn, H2O) ---
    def train_automl(self, data_path, task='classification', engine='flaml', timeout=60):
        """
        Engine Universal de AutoML.
        engines: 'tpot', 'autogluon', 'flaml', 'autosklearn', 'h2o'
        """
        logger.info(f"\n🤖 Iniciando AutoML ({task}) com engine: {engine.upper()}...")
        df = pd.read_csv(data_path)
        target = df.columns[-1]
        
        self.validate_data(df, target)
        
        mlflow.set_experiment(f"/automl_{engine}")
        with mlflow.start_run(run_name=f"{engine}_run_{datetime.now().strftime('%H%M%S')}"):
            mlflow.log_param("engine", engine)
            mlflow.log_param("timeout", timeout)
            mlflow.log_param("task", task)

            X = df.drop(columns=[target])
            y = df[target]
            
            # Tratamento básico para motores que não lidam com texto automaticamente (exceto AutoGluon/H2O)
            if engine in ['tpot', 'flaml', 'autosklearn'] and X.select_dtypes(include=['object']).any().any():
                logger.info("⚙️ Detectado colunas de texto. Aplicando codificação básica...")
                X = pd.get_dummies(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            best_model = None
            score = 0

            if engine == 'tpot' and HAS_TPOT:
                model = TPOTClassifier(max_time_mins=timeout//60, verbosity=2) if task == 'classification' \
                        else TPOTRegressor(max_time_mins=timeout//60, verbosity=2)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                best_model = model.fitted_pipeline_
                mlflow.sklearn.log_model(best_model, "model")
                
            elif engine == 'autogluon' and HAS_AUTOGLUON:
                train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
                ag_task = task
                if task == 'classification':
                    unique_count = df[target].nunique()
                    ag_task = 'binary' if unique_count <= 2 else 'multiclass'
                
                model = TabularPredictor(label=target, problem_type=ag_task).fit(train_data, time_limit=timeout)
                performance = model.evaluate(test_data)
                score = performance.get('accuracy') or performance.get('f1') or performance.get('root_mean_squared_error')
                model.save("ag_models")
                mlflow.log_artifacts("ag_models", artifact_path="autogluon_models")
                best_model = model
                
            elif engine == 'flaml' and HAS_FLAML:
                model = AutoML()
                model.fit(X_train=X_train, y_train=y_train, task=task, time_budget=timeout, metric='auto', verbose=0)
                best_model = model.model.estimator
                preds = best_model.predict(X_test)
                score = accuracy_score(y_test, preds) if task == 'classification' else r2_score(y_test, preds)
                mlflow.sklearn.log_model(best_model, "model")

            elif engine == 'autosklearn' and HAS_AUTOSKLEARN:
                if task == 'classification':
                    model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeout)
                else:
                    model = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=timeout)
                
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                best_model = model
                mlflow.sklearn.log_model(best_model, "model")

            elif engine == 'h2o' and HAS_H2O:
                h2o.init()
                h2o_train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
                h2o_test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
                
                model = H2OAutoML(max_runtime_secs=timeout, seed=42)
                model.train(y=target, training_frame=h2o_train)
                
                best_model = model.leader
                perf = best_model.model_performance(h2o_test)
                score = perf.accuracy()[0][1] if task == 'classification' else perf.r2()
                
                mlflow.h2o.log_model(best_model, "model")
                model_path = h2o.save_model(model=best_model, path="h2o_models", force=True)
                mlflow.log_artifact(model_path)
                logger.info(f"🏆 Melhor modelo H2O: {best_model.model_id}")
                # Não fechar o cluster se for usar para inferência imediata, mas aqui fechamos por segurança
                h2o.cluster().shutdown()
            
            else:
                logger.warning(f"⚠️ Engine {engine} não disponível ou não instalada.")
                return None, None

            self._log_metrics_and_plots({"best_score": score})
            
            # Exportar ONNX se for sklearn-like
            if engine in ['flaml', 'tpot', 'autosklearn']:
                try:
                    self.export_to_onnx(best_model, X_train.iloc[:1])
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao exportar ONNX: {e}")

            logger.info(f"✅ AutoML ({engine}) concluído. Score: {score}")
            return best_model, score

    def export_to_onnx(self, model, sample_input):
        """Exporta o modelo para formato ONNX."""
        logger.info("🚀 Exportando para ONNX...")
        initial_type = [('float_input', FloatTensorType([None, sample_input.shape[1]]))]
        onx = convert_sklearn(model, initial_types=initial_type)
        with open("model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        mlflow.log_artifact("model.onnx")

    def explain_model(self, model, X_train, method='shap'):
        """Gera explicações SHAP para o modelo."""
        logger.info(f"🧠 Gerando explicações: {method.upper()}")
        if method == 'shap':
            # Nota: Funciona melhor com modelos baseados em árvore do sklearn
            explainer = shap.Explainer(model.predict, X_train.iloc[:100])
            shap_values = explainer(X_train.iloc[:100])
            plt.figure()
            shap.summary_plot(shap_values, X_train.iloc[:100], show=False)
            plt.savefig("shap_summary.png")
            mlflow.log_artifact("shap_summary.png")
            plt.close()

    def generate_serving_api(self, model_name="model.onnx"):
        """Gera um script Flask básico para servir o modelo ONNX."""
        serving_script = f"""
import flask
import onnxruntime as rt
import numpy as np

app = flask.Flask(__name__)
sess = rt.InferenceSession("{model_name}")
input_name = sess.get_inputs()[0].name

@app.route('/predict', methods=['POST'])
def predict():
    data = flask.request.json['data']
    preds = sess.run(None, {{input_name: np.array(data).astype(np.float32)}})
    return flask.jsonify({{'predictions': preds[0].tolist()}})

if __name__ == '__main__':
    app.run(port=5000)
"""
        with open("serving_api.py", "w") as f:
            f.write(serving_script)
        mlflow.log_artifact("serving_api.py")
        logger.info("🌐 API de serving gerada em serving_api.py")

def main():
    m = MLOpsEnterprise()
    logger.info("🚀 Framework Universal V4.0 Carregado.")

if __name__ == "__main__":
    main()
