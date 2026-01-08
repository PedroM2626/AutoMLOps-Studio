import mlflow
from flaml import AutoML
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import json

def train_flaml_model(time_budget=10, experiment_name="flaml_experiment", task="classification"):
    """
    Treina um modelo usando FLAML e loga no MLflow.
    Suporta 'classification' e 'regression'.
    """
    print(f"Iniciando treinamento com FLAML (Task: {task})...")
    
    # 1. Preparar Dados
    if task == "classification":
        data = load_breast_cancer()
        metric = 'accuracy'
    elif task == "regression":
        data = load_diabetes()
        metric = 'r2'
    else:
        raise ValueError("Task suportada: classification, regression")

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Configurar AutoML
    automl = AutoML()
    settings = {
        "time_budget": time_budget,
        "metric": metric,
        "task": task,
        "log_file_name": 'flaml.log',
    }

    # 3. Executar Treinamento com MLflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"flaml_{task}_run") as run:
        # Logar parâmetros do FLAML
        mlflow.log_params(settings)
        
        # Treinar
        automl.fit(X_train=X_train, y_train=y_train, **settings)
        
        # Logar métricas
        print('Best ML learner:', automl.best_estimator)
        print('Best hyperparmeter config:', automl.best_config)
        
        # FLAML best_loss é minimização. 
        # Para accuracy, best_loss = 1 - acc.
        # Para r2, flaml internamente usa 1 - r2 (geralmente), mas vamos logar best_loss cru
        mlflow.log_metric("best_loss", automl.best_loss)
        
        if task == 'classification':
             print('Best accuracy on validation data: {0:.4g}'.format(1-automl.best_loss))
             mlflow.log_metric("best_validation_accuracy", 1-automl.best_loss)
        else:
             print('Best loss on validation data: {0:.4g}'.format(automl.best_loss))

        mlflow.log_metric("training_time", automl.best_config_train_time)
        
        # Logar artefatos
        mlflow.sklearn.log_model(automl, "model")
        
        # Logar config como json
        with open("best_config.json", "w") as f:
            json.dump(automl.best_config, f)
        mlflow.log_artifact("best_config.json")
        
        # Limpeza
        if os.path.exists("best_config.json"):
            os.remove("best_config.json")
        if os.path.exists("flaml.log"):
            mlflow.log_artifact("flaml.log")
            os.remove("flaml.log")

    print("Treinamento FLAML concluído.")
    return automl
