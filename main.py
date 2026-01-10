import sys
import os
# Adiciona o diretório atual ao path para importações funcionarem
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_mlflow
from src.flaml_train import train_flaml_model
from src.autogluon_train import train_autogluon_model
import argparse

def main():
    parser = argparse.ArgumentParser(description="Executar pipeline MLOps com FLAML e AutoGluon")
    parser.add_argument("--tool", type=str, choices=["flaml", "autogluon", "all"], default="all", help="Qual ferramenta executar")
    parser.add_argument("--time_budget", type=int, default=10, help="Tempo limite de treinamento em segundos (para FLAML)")
    parser.add_argument("--time_limit", type=int, default=30, help="Tempo limite de treinamento em segundos (para AutoGluon)")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], default="classification", help="Tipo de tarefa de ML")
    
    args = parser.parse_args()

    # Setup inicial
    setup_mlflow(experiment_name="mlops_demo_project")

    if args.tool in ["flaml", "all"]:
        try:
            print("\n=== Executando Pipeline FLAML ===")
            train_flaml_model(time_budget=args.time_budget, task=args.task)
        except Exception as e:
            print(f"Erro no pipeline FLAML: {e}")

    if args.tool in ["autogluon", "all"]:
        try:
            print("\n=== Executando Pipeline AutoGluon ===")
            train_autogluon_model(time_limit=args.time_limit, task=args.task)
        except Exception as e:
            print(f"Erro no pipeline AutoGluon: {e}")

if __name__ == "__main__":
    main()
