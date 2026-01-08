import mlflow
from autogluon.tabular import TabularPredictor
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import shutil

def train_autogluon_model(time_limit=30, experiment_name="autogluon_experiment", task="classification"):
    """
    Treina um modelo usando AutoGluon e loga no MLflow.
    Suporta 'classification' e 'regression'.
    """
    print(f"Iniciando treinamento com AutoGluon (Task: {task})...")

    # 1. Preparar Dados
    if task == "classification":
        data = load_breast_cancer()
        problem_type = "binary" # AutoGluon detecta, mas podemos forçar
        eval_metric = "accuracy"
    elif task == "regression":
        data = load_diabetes()
        problem_type = "regression"
        eval_metric = "r2"
    else:
        raise ValueError("Task suportada: classification, regression")

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    label = 'target'
    model_path = "autogluon_model_temp"

    # 2. Executar Treinamento com MLflow
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"autogluon_{task}_run"):
        mlflow.log_param("time_limit", time_limit)
        mlflow.log_param("label", label)
        mlflow.log_param("task", task)

        # Treinar
        # problem_type é opcional, AutoGluon infere, mas ajuda ser explícito
        predictor = TabularPredictor(label=label, problem_type=problem_type, eval_metric=eval_metric, path=model_path).fit(
            train_data, 
            time_limit=time_limit,
            presets='medium_quality' 
        )

        # Avaliar
        performance = predictor.evaluate(test_data)
        print("Performance no teste:", performance)
        
        # Logar métricas
        for metric_name, metric_val in performance.items():
            mlflow.log_metric(metric_name, metric_val)

        # Logar Leaderboard
        leaderboard = predictor.leaderboard(test_data, silent=True)
        leaderboard.to_csv("leaderboard.csv")
        mlflow.log_artifact("leaderboard.csv")

        # Logar o modelo
        mlflow.log_artifacts(model_path, artifact_path="autogluon_model")

        # Limpeza local
        if os.path.exists("leaderboard.csv"):
            os.remove("leaderboard.csv")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

    print("Treinamento AutoGluon concluído.")
    return predictor
